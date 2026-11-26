import os
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, whoami
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from safetensors.torch import load_file, load_model
from tensordict import TensorDict
from torch import nn

from .camera import decode_camera
from .conditioning import ImageEncoder, hub_repos, require_encoder_access
from .device import resolve_device
from .gaussian import Gaussian
from .models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
from .models import layers
from .models.layers import RePo3DRotaryEmbedding, UnifiedTransformerBlock
from .preprocess import CROP_MARGIN, preprocess_image
from .sampler import sample_flow

DEFAULT_REPO = 'speridlabs/idol'
DEFAULT_IMAGE_ENCODERS = {'dinov3': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
                          'flux2_vae': 'black-forest-labs/FLUX.2-dev'}
WEIGHTS_FILE = 'model.safetensors'  # the PyTorchModelHubMixin layout: one file, keys prefixed by submodule
# the layout of the first release (still the one on the Hub): one file per submodule, loaded when WEIGHTS_FILE is absent
LEGACY_WEIGHT_FILES = {'denoiser': 'denoiser.safetensors', 'decoder': 'decoder.safetensors',
                       'decoder_gs': 'decoder_gs.safetensors', 'background_remover': 'birefnet.safetensors'}

MODEL_CARD_TEMPLATE = """---
{{ card_data }}
---

# Idol: a single image to 3D Gaussians

This model has been pushed to the Hub using the
[PyTorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)
integration of [idol]({{ repo_url }}).

```bash
pip install git+{{ repo_url }}
```

```python
from idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("{{ repo_id | default("speridlabs/idol", true) }}")
output = pipeline("photo.jpg", mode="real")
output.gaussian.save_ply("photo.ply")
```

- Code: {{ repo_url }}
- Paper: {{ paper_url }}
"""


@dataclass
class IdolOutput:
    gaussian: Gaussian               # the object; .save_ply(path) writes a standard 3DGS PLY
    preprocessed_image: Image.Image  # the 1024 x 1024 image the model saw (background removed, cropped)
    camera: dict                     # predicted input camera: yaw, pitch (radians), radius, fov (degrees)


def compile_for_sampling(denoiser):
    """torch.compile the denoiser's transformer blocks and RePo layers, once per class and bound to this model's
    instances only (the decoders stay eager), and switch the RePo layers to real-valued RoPE."""
    compiled = {}
    for module in denoiser.modules():
        if isinstance(module, RePo3DRotaryEmbedding):
            module.real_valued = True
        if isinstance(module, (UnifiedTransformerBlock, RePo3DRotaryEmbedding)):
            if type(module) not in compiled:
                compiled[type(module)] = torch.compile(type(module).forward, dynamic=False)
            module.forward = types.MethodType(compiled[type(module)], module)


def encoder_sources(config: dict, overrides: Optional[dict]) -> dict:
    """config.json's image encoders with `overrides` (by name) in their place."""
    unknown = set(overrides or {}) - set(config['image_encoders'])
    if unknown:
        raise ValueError(f'image_encoders: unknown encoder {sorted(unknown)}; '
                         f'expected {sorted(config["image_encoders"])}')
    return {**config['image_encoders'], **(overrides or {})}


class IdolPipeline(nn.Module, PyTorchModelHubMixin, library_name='idol',
                   repo_url='https://github.com/speridlabs/idol-sdk', paper_url='https://speridlabs.com/research/idol',
                   license='apache-2.0', pipeline_tag='image-to-3d', tags=['gaussian-splatting', '3d', 'image-to-3d'],
                   model_card_template=MODEL_CARD_TEMPLATE):
    """Image -> 3D Gaussians. ``IdolPipeline.from_pretrained(repo_or_dir)`` loads it; the config arguments (denoiser to
    image_encoders) are the keys of the weights' config.json, the rest are runtime options that are not saved.

    optimize (CUDA, default on): torch.compile, batched CFG, TF32 and cuDNN attention (sm80+) while sampling.
    image_encoders: replace an encoder by name ('dinov3', 'flux2_vae') with a repo id, local folder or loaded model."""
    def __init__(self, denoiser: dict, decoder: dict, decoder_gs: dict, image_size: int = 1024,
                 max_voxel_level: int = 8, image_encoders: Optional[dict] = None, device: Optional[str] = None,
                 dinov3_dtype: str = 'bf16', optimize: bool = True):
        super().__init__()
        encoders = encoder_sources({'image_encoders': DEFAULT_IMAGE_ENCODERS}, image_encoders)
        # config.json holds the model config only (not device / optimize); encoders given as a local folder or a loaded
        # model are saved as their default repo
        self.config = {
            'denoiser': denoiser, 'decoder': decoder, 'decoder_gs': decoder_gs, 'image_size': image_size,
            'max_voxel_level': max_voxel_level,
            'image_encoders': {name: source if hub_repos([source]) else DEFAULT_IMAGE_ENCODERS[name]
                               for name, source in encoders.items()},
        }
        self._hub_mixin_config = self.config
        self.device = resolve_device(device)
        require_encoder_access(hub_repos([encoders['dinov3'], encoders['flux2_vae']]))

        self.denoiser = LatentSeqMMFlowModel(**denoiser['args'])
        self.decoder = OctreeProbabilityFixedlenDecoder(**decoder['args'])
        self.decoder_gs = GaussianFixedlenDecoder(**decoder_gs['args'])
        self.background_remover = BiRefNet()
        self.eval().requires_grad_(False).to(self.device)
        if self.device.type in ('cuda', 'mps'):
            self.background_remover.half()
        self.image_encoder = ImageEncoder(encoders['dinov3'], encoders['flux2_vae'], dinov3_dtype=dinov3_dtype,
                                          device=self.device)
        self.optimize = optimize and self.device.type == 'cuda'
        # PyTorch's cuDNN attention kernel needs compute capability 8.0+ (Ampere and newer); on older GPUs such as
        # the T4 (sm_75), restricting SDPA to it fails, so sampling keeps the default attention backend there.
        self.cudnn_attention = self.optimize and torch.cuda.get_device_capability(self.device)[0] >= 8
        if self.optimize:
            compile_for_sampling(self.denoiser)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str = DEFAULT_REPO, **kwargs) -> 'IdolPipeline':
        """A Hub repo id or a local folder: config.json plus model.safetensors (save_pretrained / push_to_hub), or
        the first release's per-model files (denoiser, decoder, decoder_gs, birefnet .safetensors).

        kwargs: device, dinov3_dtype, optimize, image_encoders (see the class) and the Hub download options of
        PyTorchModelHubMixin.from_pretrained (revision, token, cache_dir, force_download, local_files_only)."""
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def _from_pretrained(cls, *, model_id: str, revision: Optional[str], cache_dir, force_download: bool,
                         local_files_only: bool, token, config: Optional[dict] = None, strict: bool = True,
                         **model_kwargs) -> 'IdolPipeline':
        if config is None:
            raise FileNotFoundError(f'{model_id}: no config.json, not Idol weights')
        # image_encoders overrides a few encoders by name; the others stay the ones in config.json
        model_kwargs['image_encoders'] = encoder_sources(config, model_kwargs.get('image_encoders'))
        model = cls(**model_kwargs)

        def fetch(filename):
            if os.path.isdir(model_id):
                path = os.path.join(model_id, filename)
                if not os.path.isfile(path):
                    raise EntryNotFoundError(f'{path} not found')
                return path
            return hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir,
                                   force_download=force_download, local_files_only=local_files_only, token=token)

        try:
            load_model(model, fetch(WEIGHTS_FILE), strict=strict)
        except EntryNotFoundError:  # (also offline, when not cached) the first release's per-model files instead
            for name, filename in LEGACY_WEIGHT_FILES.items():
                getattr(model, name).load_state_dict(load_file(fetch(filename)), strict=True)
        return model.eval()

    def push_to_hub(self, repo_id: str, *, model_card_kwargs: Optional[dict] = None, **kwargs) -> str:
        """PyTorchModelHubMixin.push_to_hub; a bare repo name goes to your account and the model card shows the repo."""
        if '/' not in repo_id:
            repo_id = f"{whoami(token=kwargs.get('token'))['name']}/{repo_id}"
        model_card_kwargs = {'repo_id': repo_id, **(model_card_kwargs or {})}
        return super().push_to_hub(repo_id, model_card_kwargs=model_card_kwargs, **kwargs)

    def preprocess(self, image: Union[str, Image.Image], mode: str = 'synthetic',
                   crop_margin: Optional[float] = None) -> Image.Image:
        """mode 'synthetic' (renders, generated images of one object: crop 1.2) or 'real' (photos: crop 1.7);
        crop_margin overrides the mode's value."""
        if isinstance(image, str):
            image = Image.open(image)
        margin = crop_margin if crop_margin is not None else CROP_MARGIN[mode]
        return preprocess_image(image, background_remover=self.background_remover,
                                image_size=self.config['image_size'], crop_margin=margin)

    @torch.no_grad()
    def forward(
        self,
        image: Union[str, Image.Image],
        mode: str = 'synthetic',
        seed: int = 42,
        steps: int = 50,
        cfg_strength: float = 3.0,
        rescale_t: float = 3.0,
        num_gaussians: int = 262144,
        crop_margin: Optional[float] = None,
        verbose: bool = False,
    ) -> IdolOutput:
        """Generate the 3D Gaussians of the object in ``image``.

        mode / crop_margin: see ``preprocess``.
        num_gaussians: 32,768 to 1,048,576, rounded down to a multiple of 32."""
        prepared = self.preprocess(image, mode=mode, crop_margin=crop_margin)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(self.device)

        torch.manual_seed(seed)
        cond = self.image_encoder(pixels)
        noise = TensorDict({
            'latent': torch.randn(1, self.denoiser.q_token_length, self.denoiser.in_channels, device=self.device),
            'camera': torch.randn(1, 1, self.denoiser.cam_channels, device=self.device),
        }, batch_size=1, device=self.device)
        with self._sampling_settings():
            sample = sample_flow(self.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                                 steps=steps, rescale_t=rescale_t, cfg_strength=cfg_strength, verbose=verbose,
                                 batch_cfg=self.optimize)
        gaussian = self.decode(sample['latent'], num_gaussians)
        return IdolOutput(gaussian=gaussian, preprocessed_image=prepared, camera=decode_camera(sample['camera'][0, 0]))

    @contextmanager
    def _sampling_settings(self):
        """optimize=True: cuDNN attention (where supported) and TF32 matmuls for the denoiser only (cuDNN is slower in
        the decoders)."""
        if not self.optimize:
            yield
            return
        backend, tf32 = layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
        if self.cudnn_attention:
            layers.ATTN_BACKEND = 'sdpa_cudnn'
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            layers.ATTN_BACKEND = backend
            torch.backends.cuda.matmul.allow_tf32 = tf32

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, num_gaussians: int = 262144) -> Gaussian:
        """Latent [1, 8192, 16] -> Gaussians: the octree decoder places num_gaussians / 32 anchor points, the
        Gaussian decoder turns each anchor into 32 Gaussians."""
        per_anchor = self.decoder_gs.rep_config['num_gaussians']
        anchors = OctreeProbabilityFixedlenDecoder.sample(self.decoder, latent,
                                                          num_points=max(1, int(num_gaussians) // per_anchor),
                                                          level=self.config['max_voxel_level'])
        return self.decoder_gs.to_representation(anchors, self.decoder_gs(anchors, latent))[0]
