"""Fast Idol pipeline on CUDA (Python + idol_cuda kernels).

Front end (preprocess, BiRefNet, DINOv3, FLUX.2 VAE) is the reference code; the RNG is consumed in the same order
as the reference (seed, VAE posterior noise, latent noise, camera noise), so the sampler starts from the same noise.
Then: fast sampler (fast/model.py), octree descent with a loop-free systematic resampling, the reference Gaussian
decoder, and the Gaussian activations + PLY columns fused in one kernel (`gaussians_ply`), written with one write.

parity=True turns every arithmetic-changing piece off: reference sampler structure with only bit-equal reuse
(context once, shared per-step prefix), reference octree sampling, reference activations + plyfile writer.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from idol import IdolPipeline
from idol.camera import decode_camera
from idol.models import OctreeProbabilityFixedlenDecoder

from .model import FastDenoiser
from .ops import kernel

PLY_NAMES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def systematic_sample_fast(probs, counts):
    """systematic_sample without the per-count Python loop: for row r with n = counts[r] points and offset
    u0 = U/n, bin j gets #{k < n : cdf[j-1] < u0 + k/n <= cdf[j]} = F(cdf[j]) - F(cdf[j-1]),
    F(c) = clamp(floor((c - u0) n) + 1, 0, n); the last bin takes the remainder (as clamp_max(P-1) does).
    One torch.rand(R) draw per level instead of one per distinct count (different random numbers, same law)."""
    R, P = probs.shape
    probs = probs.float().clamp_min(0)
    s = probs.sum(1, keepdim=True)
    probs = torch.where(s.eq(0), torch.full_like(probs, 1.0 / P), probs / s.clamp_min(1))
    cdf = probs.cumsum(1)
    n = counts.float().unsqueeze(1)
    u0 = torch.rand(R, 1, device=probs.device) / n.clamp_min(1)
    Fc = torch.clamp(torch.floor((cdf[:, :-1] - u0) * n) + 1, min=0)
    Fc = torch.minimum(Fc, n)
    Fc = torch.cat([torch.zeros_like(n), Fc, n], 1)
    return (Fc[:, 1:] - Fc[:, :-1]).long()


@torch.no_grad()
def octree_sample_fast(model, cond, num_points, level):
    """OctreeProbabilityFixedlenDecoder.sample for one object, without per-level Python loops over counts."""
    dev = cond.device
    child_offset = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]], dtype=torch.long,
                                device=dev)
    coords = torch.zeros(1, 3, dtype=torch.long, device=dev)
    counts = torch.full((1,), num_points, dtype=torch.long, device=dev)
    for lv in range(1, level + 1):
        res_p = 1 << (lv - 1)
        parent = ((coords.float() + 0.5) / res_p).unsqueeze(0)
        res = torch.full((1,), 1 << lv, dtype=torch.long, device=dev)
        probs = torch.softmax(model(parent, res, cond), dim=-1)[0]
        sampled = systematic_sample_fast(probs, counts).flatten()
        child = (coords[:, None, :] * 2 + child_offset[None]).flatten(0, 1)
        keep = sampled > 0
        coords, counts = child[keep], sampled[keep]
    pts = torch.repeat_interleave(coords, counts, dim=0, output_size=num_points)
    return ((pts.float() + torch.rand(pts.shape, device=dev)) / (1 << level)).unsqueeze(0)


def write_ply_rows(path, rows):
    """rows [N, 17] float32 (host) -> the binary PLY plyfile writes for Gaussian.save_ply (same header, same bytes)."""
    rows = np.ascontiguousarray(rows, dtype='<f4')
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % rows.shape[0]
    header += ''.join(f'property float {n}\n' for n in PLY_NAMES) + 'end_header\n'
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rows.tobytes())


class FastOutput:
    def __init__(self, rows=None, gaussian=None, preprocessed_image=None, camera=None, latent=None):
        self.rows, self.gaussian, self.preprocessed_image, self.camera, self.latent = (rows, gaussian,
                                                                                       preprocessed_image, camera,
                                                                                       latent)

    def save_ply(self, path):
        if self.rows is not None:
            write_ply_rows(path, self.rows.cpu().numpy())
        else:
            self.gaussian.save_ply(path)


class FastIdolPipeline:
    def __init__(self, weights='/w/idol', dinov3='/w/dinov3', flux2='/w/flux2', parity=False, collapse_uncond=True,
                 attn='fa3', cuda_graph=False, fast_octree=True, fast_export=True, ref=None):
        """ref: an already loaded IdolPipeline(optimize=False) to build on (its modules are shared, not modified),
        e.g. so one process can run the official eager pipeline and the fast build on the same weights."""
        os.environ.setdefault('ATTN_BACKEND', 'sdpa')
        self.parity = parity
        if ref is None:
            enc = {k: v for k, v in (('dinov3', dinov3), ('flux2_vae', flux2)) if v}  # None: config.json's Hub ids
            ref = IdolPipeline(weights, device='cuda', optimize=False, image_encoders=enc or None)
        assert not ref.optimize, 'FastIdolPipeline builds on the eager reference (optimize=False)'
        self.ref = ref
        self.fast_octree = fast_octree and not parity
        self.fast_export = fast_export and not parity
        if not parity:
            self.sampler = FastDenoiser(self.ref.denoiser, collapse_uncond=collapse_uncond, attn=attn,
                                        cuda_graph=cuda_graph)
            self.K = kernel()
            g = self.ref.decoder_gs
            rc = g.rep_config
            x = torch.tensor(rc['scaling_bias'])
            self.scaling_offset = float(x + torch.log(-torch.expm1(-x)))
            y = torch.tensor(rc['opacity_bias'])
            self.opacity_offset = float(torch.log(y / (1 - y)))
            self.gs_args = (float(g.base_offset_scale), self.scaling_offset, self.opacity_offset,
                            rc['3d_filter_kernel_size'], rc['lr']['_rotation'], rc['perturbe_size'])
            for k in ('_xyz', '_features_dc', '_scaling', '_opacity'):
                assert rc['lr'][k] == 1.0, 'gaussians_ply assumes lr 1.0 except rotation'
            assert rc['num_gaussians'] == 32

    @torch.no_grad()
    def __call__(self, image, mode='synthetic', seed=42, steps=50, cfg_strength=3.0, rescale_t=3.0,
                 num_gaussians=262144, timings=None):
        ref = self.ref

        def mark(name):
            if timings is not None:
                torch.cuda.synchronize()
                timings.append((name, __import__('time').perf_counter()))
        mark('start')
        prepared = ref.preprocess(image, mode=mode)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(ref.device)
        mark('preprocess')
        torch.manual_seed(seed)
        cond = ref.image_encoder(pixels)
        mark('encoders')
        den = ref.denoiser
        noise = TensorDict({
            'latent': torch.randn(1, den.q_token_length, den.in_channels, device=ref.device),
            'camera': torch.randn(1, 1, den.cam_channels, device=ref.device),
        }, batch_size=1, device=ref.device)
        if self.parity:
            from .parity import sample_parity
            z, cam = sample_parity(den, noise, cond, steps, rescale_t, cfg_strength)
        else:
            z, cam = self.sampler.sample(noise, cond, steps=steps, rescale_t=rescale_t, cfg=cfg_strength)
        mark('sampling')
        per_anchor = ref.decoder_gs.rep_config['num_gaussians']
        npts = max(1, int(num_gaussians) // per_anchor)
        if self.fast_octree:
            anchors = octree_sample_fast(ref.decoder, z, npts, ref.config['max_voxel_level'])
        else:
            anchors = OctreeProbabilityFixedlenDecoder.sample(ref.decoder, z, num_points=npts,
                                                              level=ref.config['max_voxel_level'])
        mark('octree')
        feats = ref.decoder_gs(anchors, z)
        mark('gs_decoder')
        camera = decode_camera(cam[0, 0])
        if self.fast_export:
            rows = self.K.gaussians_ply(feats[0].contiguous(), anchors[0].contiguous(),
                                        ref.decoder_gs.points_offset_perturbation.contiguous(), *self.gs_args)
            out = FastOutput(rows=rows, preprocessed_image=prepared, camera=camera, latent=z)
        else:
            g = ref.decoder_gs.to_representation(anchors, feats)[0]
            out = FastOutput(gaussian=g, preprocessed_image=prepared, camera=camera, latent=z)
        mark('to_gaussians')
        return out


def save_outputs(out, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    out.save_ply(os.path.join(outdir, f'{name}.ply'))
    out.preprocessed_image.save(os.path.join(outdir, f'{name}_input.png'))
    with open(os.path.join(outdir, f'{name}_camera.json'), 'w') as f:
        json.dump(out.camera, f, indent=2)
