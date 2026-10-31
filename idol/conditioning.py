import torch
import torch.nn.functional as F
from tensordict import TensorDict

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
_ACCESSIBLE_REPOS = set()


class EncoderAccessError(RuntimeError):
    """The Hugging Face account in use cannot download one of the gated image encoders."""


def require_encoder_access(repos):
    """Stop with an explanation, before anything large is downloaded, when the gated encoder repos are not readable.

    Network problems are not treated as missing access: the encoders may already be in the local cache."""
    import os

    from huggingface_hub import auth_check, get_token
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if os.environ.get('HF_HUB_OFFLINE', '0').lower() in ('1', 'true', 'yes'):
        return
    denied = []
    for repo in repos:
        if repo in _ACCESSIBLE_REPOS:
            continue
        try:
            auth_check(repo)
            _ACCESSIBLE_REPOS.add(repo)
        except (GatedRepoError, RepositoryNotFoundError):
            denied.append(repo)
        except Exception:
            pass
    if not denied:
        return
    pages = '\n'.join(f'  https://huggingface.co/{repo}' for repo in denied)
    if get_token() is None:
        reason = 'You are not logged in to Hugging Face, and these image encoders are gated:'
        steps = ('1. Accept the terms on each page above with your Hugging Face account.\n'
                 '2. Log in: run `hf auth login`, or set the HF_TOKEN environment variable to a token of that account.')
    else:
        reason = 'Your Hugging Face account does not have access yet to these gated image encoders:'
        steps = ('1. Accept the terms on each page above (approval can take a while).\n'
                 '2. Make sure the token in use (`hf auth whoami`) belongs to the account that accepted them.')
    raise EncoderAccessError(f'Idol needs two image encoders that are downloaded from Hugging Face.\n{reason}\n'
                             f'{pages}\n{steps}')


def hub_repos(sources):
    """The encoder sources that would be downloaded from the Hub: strings that are not local directories."""
    import os

    return [s for s in sources if isinstance(s, str) and not os.path.isdir(s)]


class ImageEncoder:
    """dinov3 and flux2_vae are each a Hub repo id, a local directory in the same layout (the FLUX.2 one with its
    autoencoder under vae/), or an already-loaded model (a transformers DINOv3 model, an AutoencoderKLFlux2), used in the dtype it was loaded in
    (load it as from_pretrained would: DINOv3 in dinov3_dtype, the autoencoder in bf16; a blanket cast would also cast
    the buffers transformers keeps in fp32, such as the RoPE frequencies)."""
    def __init__(self, dinov3, flux2_vae, dinov3_dtype: str = 'bf16', device: str = 'cuda'):
        from diffusers.models import AutoencoderKLFlux2
        from transformers import AutoModel

        self.device = torch.device(device)
        if isinstance(dinov3, str):
            dinov3 = AutoModel.from_pretrained(dinov3, torch_dtype=DTYPES[dinov3_dtype])
        self.dinov3 = dinov3.eval().requires_grad_(False).to(self.device)
        if isinstance(flux2_vae, str):
            flux2_vae = AutoencoderKLFlux2.from_pretrained(flux2_vae, subfolder='vae', torch_dtype=torch.bfloat16)
        self.flux2_vae = flux2_vae.eval().requires_grad_(False).to(self.device)

    def _dinov3_tokens(self, images):
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images.float() - mean) / std
        dtype = next(self.dinov3.parameters()).dtype
        tokens = self.dinov3(pixel_values=images.to(dtype)).last_hidden_state.float()
        return F.layer_norm(tokens, tokens.shape[-1:])

    def _flux2_tokens(self, images):
        vae = self.flux2_vae
        latents = vae.encode(images.to(torch.bfloat16) * 2 - 1).latent_dist.sample()
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
        padding = torch.zeros(batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
        return torch.cat([padding, tokens], dim=1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> TensorDict:
        """images: [B, 3, 1024, 1024] in [0, 1] -> TensorDict(feature1=DINOv3 tokens, feature2=FLUX.2 VAE tokens)."""
        images = images.to(self.device)
        feature1 = self._dinov3_tokens(images)
        feature2 = self._flux2_tokens(images)
        return TensorDict({'feature1': feature1, 'feature2': feature2}, batch_size=feature1.shape[0])
