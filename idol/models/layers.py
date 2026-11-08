import importlib.util
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# True when PyTorch is built for AMD GPUs (ROCm/HIP), where the CUDA-only flash-attn package and the cuDNN
# attention kernel are unavailable.
IS_ROCM = getattr(torch.version, 'hip', None) is not None

# flash-attn when it is installed, PyTorch SDPA otherwise (same result, flash-attn is faster). flash-attn is
# CUDA-only, so on ROCm SDPA is used (the dispatcher selects the fused AOTriton flash-attention kernel).
# Override with ATTN_BACKEND=flash_attn|sdpa|sdpa_cudnn (sdpa_cudnn: PyTorch SDPA restricted to the cuDNN kernel,
# which IdolPipeline(optimize=True) selects for the denoiser while sampling; see FAST_SDPA_BACKEND).
ATTN_BACKEND = os.environ.get(
    'ATTN_BACKEND',
    'flash_attn' if not IS_ROCM and importlib.util.find_spec('flash_attn') is not None else 'sdpa')
if ATTN_BACKEND not in ('flash_attn', 'sdpa', 'sdpa_cudnn'):
    raise ValueError(f'Unknown attention backend: {ATTN_BACKEND}')
if ATTN_BACKEND == 'flash_attn':
    import flash_attn

# The fastest restricted SDPA for the optimized sampling path (IdolPipeline._sampling_settings): the cuDNN
# kernel on NVIDIA GPUs. On ROCm no restriction is needed: plain SDPA already selects the fused AOTriton
# flash-attention kernel for half-precision inputs, and restricting to CUDNN_ATTENTION would fail.
FAST_SDPA_BACKEND = 'sdpa' if IS_ROCM else 'sdpa_cudnn'


def scaled_dot_product_attention(*args) -> torch.Tensor:
    """(qkv [N, L, 3, H, C]) or (q [N, L, H, C], k [N, S, H, C], v [N, S, H, C]) -> [N, L, H, C]."""
    if ATTN_BACKEND == 'flash_attn':
        if len(args) == 1:
            return flash_attn.flash_attn_qkvpacked_func(args[0])
        return flash_attn.flash_attn_func(*args)
    q, k, v = args[0].unbind(dim=2) if len(args) == 1 else args
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    if ATTN_BACKEND == 'sdpa_cudnn' and not IS_ROCM:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3)


def convert_module_to_f16(module):
    if isinstance(module, nn.Linear):
        for parameter in module.parameters():
            parameter.data = parameter.data.half()


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, channels: int, inner_channels: int, channels_out: int, mlp_layer_num: int = 2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(nn.Linear(channels if i == 0 else inner_channels, inner_channels))
            layers.append(nn.GELU(approximate="tanh"))
        layers.append(nn.Linear(inner_channels, channels_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -- Positional and scalar embeddings ----------------------------------------

class PcdAbsolutePositionEmbedder(nn.Module):
    """Sinusoidal embedding of 3D positions in [0, 1] (the denoiser's latent token positions)."""
    def __init__(self, channels: int, in_channels: int = 3, max_res=16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs_2exp = torch.arange(max_res, dtype=torch.float32)
        res_dim = max(0, self.freq_dim - max_res)
        freqs_res = torch.arange(res_dim, dtype=torch.float32) / res_dim * max_res
        self.freqs = 2 ** torch.cat([freqs_2exp, freqs_res], dim=0)[:self.freq_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *dims, _ = x.shape
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x.reshape(-1), self.freqs) * 2 * torch.pi
        embed = torch.cat([torch.sin(out), torch.cos(out)], dim=-1).reshape(*dims, -1)
        if embed.shape[-1] < self.channels:
            embed = torch.cat([embed, torch.zeros(*dims, self.channels - embed.shape[-1], device=embed.device)], dim=-1)
        return embed


class PcdAbsolutePositionEmbedderv2(nn.Module):
    """Sinusoidal embedding of 3D points in [0, 1] (the decoders' query points)."""
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.freq_dim = channels // in_channels // 2
        logs = torch.linspace(0.0, float(max_res), steps=self.freq_dim, dtype=torch.float32)
        self.register_buffer("freqs", torch.pow(2.0, logs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        freqs = self.freqs.to(x.device, dtype=x.dtype)
        ang = x.unsqueeze(-1) * freqs * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device,
                                                  dtype=embed.dtype)], dim=-1)
        return embed


class SinusoidalEmbedder(nn.Module):
    """Embeds a scalar per batch element (diffusion time, octree level) into a vector."""
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, two_pi=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.two_pi = two_pi

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        if self.two_pi:
            args = args * 2 * torch.pi
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """hidden_states [B, L, H, D] rotated by the complex frequencies freqs [B, L, H or 1, D/2], or by the same
    rotation as real (cos, sin) pairs [B, L, H, D/2, 2]."""
    if not freqs.is_complex():
        # the complex product written out, so torch.compile can fuse it
        x0, x1 = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
        cos, sin = freqs.unbind(-1)
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2).type_as(hidden_states)
    x_rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    x_rotated = x_rotated * freqs
    x_out = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1)
    return x_out.type_as(hidden_states)


def clamp_mul(x, f):
    f_t = f.tanh()
    return x * f_t + x.detach() * (f - f_t)


class RePo3DRotaryEmbedding(nn.Module):
    """Learned rotary position: each token's features predict a 3D offset per head, turned into RoPE angles."""
    # True: return (cos, sin) pairs instead of a complex tensor, which torch.compile cannot fuse (optimize=True sets it)
    real_valued = False

    def __init__(self, model_channels: int, num_heads: int, head_dim: int, repo_hidden_ratio: float = 0.125,
                 max_freq: float = 16.0):
        super().__init__()
        self.num_heads = num_heads
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        # head_dim split into 3 parts for x, y, z
        dim_0 = dim_1 = 2 * (head_dim // 6)
        dims = [dim_0, dim_1, head_dim - dim_0 - dim_1]
        freqs = [torch.linspace(1.0, float(max_freq), steps=d // 2, dtype=torch.float32) for d in dims]
        self.freqs_0 = nn.Parameter(freqs[0])
        self.freqs_1 = nn.Parameter(freqs[1])
        self.freqs_2 = nn.Parameter(freqs[2])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states.float())
        out = self.final_map(self.act(self.gate_map(h)) * self.content_map(h))
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1)
        if self.real_valued:
            return torch.stack([ang.cos(), ang.sin()], dim=-1)
        return torch.polar(torch.ones_like(ang), ang).type(torch.complex64)


# -- Attention ---------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Self or cross attention with RMS-normalised queries and keys (the decoders)."""
    def __init__(self, channels: int, num_heads: int, ctx_channels: Optional[int] = None, cross: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.cross = cross
        head_dim = channels // num_heads
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self.cross:
            q = self.q_rms_norm(self.to_q(x).reshape(B, L, self.num_heads, -1))
            k, v = self.to_kv(context).reshape(B, context.shape[1], 2, self.num_heads, -1).unbind(dim=2)
            h = scaled_dot_product_attention(q, self.k_rms_norm(k), v)
        else:
            q, k, v = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1).unbind(dim=2)
            h = scaled_dot_product_attention(torch.stack([self.q_rms_norm(q), self.k_rms_norm(k), v], dim=2))
        return self.to_out(h.reshape(B, L, -1))


class RopeMultiHeadAttention(nn.Module):
    """Self attention with rotary embeddings and RMS-normalised queries and keys (the denoiser)."""
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, rope_emb: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).unbind(2)
        q = self.q_norm(apply_rotary_emb(q, rope_emb))
        k = self.k_norm(apply_rotary_emb(k, rope_emb))
        return self.out(scaled_dot_product_attention(q, k, v).reshape(B, L, C))


# -- Transformer blocks ------------------------------------------------------

class UnifiedTransformerBlock(nn.Module):
    """Denoiser block: RoPE self attention + FFN, modulated by the diffusion time through a shared adaLN plus
    this block's own shift table (modulation=False for the condition refiner, which has plain LayerNorms)."""
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0, modulation: bool = True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if modulation:
            self.shift_table = nn.Parameter(torch.zeros(1, 6 * channels))

    def forward(self, x: torch.Tensor, mod: Optional[torch.Tensor], rotary_emb: torch.Tensor) -> torch.Tensor:
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            return x + self.mlp(self.norm2(x))
        mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, rope_emb=rotary_emb)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return x + h * gate_mlp.unsqueeze(1)


class TransformerCrossBlock(nn.Module):
    """Gaussian decoder block: self attention + cross attention to the latent + FFN."""
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(channels, num_heads=num_heads)
        self.cross_attn = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """Octree decoder block: cross attention to the latent + FFN, modulated by the octree level (shared adaLN)."""
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.cross_attn(h, context)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return x + h * gate_mlp.unsqueeze(1)
