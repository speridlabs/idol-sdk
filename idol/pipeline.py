import json
import os
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from tensordict import TensorDict

from .camera import decode_camera
from .conditioning import ImageEncoder, hub_repos, require_encoder_access
from .gaussian import Gaussian
from .models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
from .models import layers
from .models.layers import RePo3DRotaryEmbedding, UnifiedTransformerBlock
from .preprocess import CROP_MARGIN, preprocess_image
from .sampler import sample_flow

DEFAULT_REPO = 'speridlabs/idol'
WEIGHT_FILES = ['config.json', 'denoiser.safetensors', 'decoder.safetensors', 'decoder_gs.safetensors',
                'birefnet.safetensors']


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


class IdolPipeline:
    """optimize (default on, CUDA only): ~1.7x faster sampling at the same output quality: compiled denoiser
    (torch.compile), conditional and unconditional passes batched together, cuDNN attention and TF32 matmuls while
    sampling. The first generation in a process compiles first. optimize=False runs everything eagerly.

    image_encoders overrides where the image encoders come from, by name ('dinov3', 'flux2_vae'; default: the repos
    in config.json): a Hub repo id, a local directory, or an already-loaded model with the same weights. Only Hub repo
    ids are checked for gated access."""
    def __init__(self, weights_dir: str, device: Optional[str] = None, dinov3_dtype: str = 'bf16',
                 optimize: bool = True, image_encoders: Optional[dict] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        with open(os.path.join(weights_dir, 'config.json')) as f:
            self.config = json.load(f)
        encoders = encoder_sources(self.config, image_encoders)
        require_encoder_access(hub_repos([encoders['dinov3'], encoders['flux2_vae']]))

        def load(model, name):
            model.load_state_dict(load_file(os.path.join(weights_dir, f'{name}.safetensors')), strict=True)
            return model.eval().requires_grad_(False).to(self.device)

        self.denoiser = load(LatentSeqMMFlowModel(**self.config['denoiser']['args']), 'denoiser')
        self.decoder = load(OctreeProbabilityFixedlenDecoder(**self.config['decoder']['args']), 'decoder')
        self.decoder_gs = load(GaussianFixedlenDecoder(**self.config['decoder_gs']['args']), 'decoder_gs')
        self.background_remover = load(BiRefNet(), 'birefnet')
        if self.device.type == 'cuda':
            self.background_remover.half()
        self.image_encoder = ImageEncoder(encoders['dinov3'], encoders['flux2_vae'], dinov3_dtype=dinov3_dtype,
                                          device=self.device)
        self.optimize = optimize and self.device.type == 'cuda'
        if self.optimize:
            compile_for_sampling(self.denoiser)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = DEFAULT_REPO, **kwargs) -> 'IdolPipeline':
        if os.path.isdir(repo_or_dir):
            return cls(repo_or_dir, **kwargs)
        from huggingface_hub import hf_hub_download, snapshot_download
        # check the gated image encoders before downloading the ~3 GB of weights
        with open(hf_hub_download(repo_or_dir, 'config.json')) as f:
            encoders = encoder_sources(json.load(f), kwargs.get('image_encoders'))
        require_encoder_access(hub_repos([encoders['dinov3'], encoders['flux2_vae']]))
        return cls(snapshot_download(repo_or_dir, allow_patterns=WEIGHT_FILES), **kwargs)

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
    def __call__(
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
        """optimize=True: cuDNN attention and TF32 matmuls for the denoiser only (cuDNN is slower in the decoders)."""
        if not self.optimize:
            yield
            return
        backend, tf32 = layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
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
