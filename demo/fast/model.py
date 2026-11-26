"""Fast denoiser for Idol on CUDA (the fast build's sampler), built from the reference LatentSeqMMFlowModel's weights.

Structure (see docs/ab-log.md for each change and its measured speed and error):
  * the context stream (cond embedders + context refiner) depends only on the image: computed once per request;
  * time embedding, adaLN, noise refiner and camera refiner are shared by the cond and uncond passes (same input);
  * cond and uncond rows are packed into one [12294 + 8194, 1024] matrix, so every GEMM and row kernel runs once;
  * collapse_uncond: the unconditional context is 4,101 identical tokens (zero features): it is carried as ONE token
    whose key counts 4,101 times (attention over the other keys + an exact log-sum-exp merge of that key);
  * fused kernels (idol_cuda): residual + LayerNorm32 + adaLN modulation + the next RePo LayerNorm in one pass;
    RePo rotary + head RMSNorm for q and k in one pass; fc1 + bias + GELU in the cuBLASLt epilogue;
  * attention: FlashAttention-3 (Hopper, kernels-community/flash-attn3 from the Hub), cuDNN as fallback;
    RePo fp32 GEMMs in TF32 (as shipped);
  * optional CUDA graph over one whole step.
"""
import math

import torch
import torch.nn.functional as F

from .ops import kernel

C, H, D = 1024, 16, 64


def _attn(q, k, v, backend, need_lse):
    """q [1, Lq, H, D], k/v [1, Lk, H, D] (strided views) -> out [1, Lq, H, D], lse [1, H, Lq] or None."""
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if backend == 'cudnn':
        r = torch.ops.aten._scaled_dot_product_cudnn_attention(qt, kt, vt, None, need_lse, 0.0, False, False)
        o, lse = r[0], r[1]
    elif backend == 'flash':
        r = torch.ops.aten._scaled_dot_product_flash_attention(qt, kt, vt, 0.0, False, False)
        o, lse = r[0], r[1]
    elif backend in ('fa3', 'fa3_fp8'):
        fa3 = _fa3()
        if backend == 'fa3_fp8':
            q8, k8, v8 = (t.to(torch.float8_e4m3fn) for t in (q, k, v))
            r = fa3.flash_attn_func(q8, k8, v8, return_attn_probs=True)
        else:
            r = fa3.flash_attn_func(q, k, v, return_attn_probs=True)
        o, lse = (r[0], r[1]) if isinstance(r, (tuple, list)) else (r, None)
        return o.to(torch.float16), (lse if need_lse else None)
    elif backend == 'fa2':
        import flash_attn
        o, lse, _ = flash_attn.flash_attn_func(q, k, v, return_attn_probs=True)
        return o, (lse if need_lse else None)
    else:
        raise ValueError(backend)
    o = o.transpose(1, 2)
    if need_lse:
        lse = lse.reshape(lse.shape[0], lse.shape[1], lse.shape[2])
    return o, (lse if need_lse else None)


_FA3 = None


def _fa3():
    """FlashAttention-3 (Hopper) from the Hub's kernels-community/flash-attn3 (downloaded once)."""
    global _FA3
    if _FA3 is None:
        import os
        from kernels import get_kernel
        off = os.environ.pop('HF_HUB_OFFLINE', None)
        try:
            _FA3 = get_kernel('kernels-community/flash-attn3', version=int(os.environ.get('FA3_VERSION', '1')))
        finally:
            if off is not None:
                os.environ['HF_HUB_OFFLINE'] = off
    return _FA3


class FastDenoiser:
    def __init__(self, den, collapse_uncond=True, attn='fa3', tf32=True, cuda_graph=False):
        self.K = kernel()
        if attn == 'fa3':
            try:
                _fa3()
            except Exception as e:  # no Hub access / no Hopper build: cuDNN (the shipped optimize path's kernel)
                print(f'FlashAttention-3 unavailable ({type(e).__name__}: {e}); using cuDNN attention', flush=True)
                attn = 'cudnn'
        self.den = den
        self.collapse = collapse_uncond
        self.attn = attn
        self.tf32 = tf32
        self.cuda_graph = cuda_graph
        self.q_len = den.q_token_length
        dev = den.pos_pe.device
        self.dev = dev
        with torch.no_grad():
            self.pos_embed = den.pos_embedder(den.pos_pe).type(torch.float16)[0].contiguous()  # [8192, 1024]

        def repo(r):
            return dict(w=r.norm.weight.float().contiguous(), b=r.norm.bias.float().contiguous(),
                        gc=torch.cat([r.gate_map.weight, r.content_map.weight], 0).float().contiguous(),
                        fin=r.final_map.weight.float().contiguous(),
                        freqs=torch.cat([r.freqs_0, r.freqs_1, r.freqs_2]).float().contiguous())

        def block(b):
            a = b.attn
            d = dict(qkv_w=a.qkv.weight, qkv_b=a.qkv.bias, out_w=a.out.weight, out_b=a.out.bias,
                     gq=a.q_norm.gamma.float().contiguous(), gk=a.k_norm.gamma.float().contiguous(),
                     fc1_w=b.mlp.mlp[0].weight, fc1_b=b.mlp.mlp[0].bias, fc2_w=b.mlp.mlp[2].weight,
                     fc2_b=b.mlp.mlp[2].bias, modulation=b.modulation)
            if b.modulation:
                d['shift_table'] = b.shift_table
            else:
                d.update(n1w=b.norm1.weight.float(), n1b=b.norm1.bias.float(), n2w=b.norm2.weight.float(),
                         n2b=b.norm2.bias.float())
            return d
        self.noise = [(repo(r), block(b)) for r, b in zip(den.noise_repo_layers, den.noise_refiner)]
        self.ctx = [(repo(r), block(b)) for r, b in zip(den.context_repo_layers, den.context_refiner)]
        self.joint = [(repo(r), block(b)) for r, b in zip(den.repo_layers, den.blocks)]
        self.graph = None
        # SinusoidalEmbedder's frequencies, computed on the CPU exactly as the reference, kept on the device
        import numpy as np
        half = den.t_embedder.frequency_embedding_size // 2
        self.t_freqs = torch.exp(-np.log(den.t_embedder.max_period) *
                                 torch.arange(start=0, end=half, dtype=torch.float32) / half).to(dev)

    def _t_embed(self, t):
        args = t[:, None].float() * self.t_freqs[None]
        return self.den.t_embedder.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

    # ------------------------------------------------------------------------------------------ pieces
    def _delta(self, r, ln):
        """RePo offsets [M, 48] from the fp32 LayerNorm output (gate/content maps fused into one TF32 GEMM)."""
        gc = ln @ r['gc'].t()
        return (F.silu(gc[:, :128]) * gc[:, 128:]) @ r['fin'].t()

    def _attention(self, qkv, delta, r, b, segs):
        """qkv [M, 3072] fp16 -> attention output [M, 1024] fp16 for the row segments
        segs = [(start, length, extra_key_row or None, log_mult)]."""
        K = self.K
        M = qkv.shape[0]
        q, k = K.rope_rms_qk(qkv, delta, r['freqs'], b['gq'], b['gk'])
        v = qkv[:, 2 * C:].view(M, H, D)
        out = torch.empty(M, H, D, device=qkv.device, dtype=torch.float16)
        for (s, n, extra, log_mult) in segs:
            qs = q[s:s + n].unsqueeze(0)
            if n == 1 and extra is None:   # softmax over one key is exactly 1: the output is v
                out[s:s + 1].copy_(v[s:s + 1])
            elif extra is None:
                o, _ = _attn(qs, k[s:s + n].unsqueeze(0), v[s:s + n].unsqueeze(0), self.attn, False)
                out[s:s + n].copy_(o[0])
            else:
                # keys: the segment minus its last row; the last row is the collapsed context token (multiplicity)
                o, lse = _attn(qs, k[s:s + n - 1].unsqueeze(0), v[s:s + n - 1].unsqueeze(0), self.attn, True)
                if o.stride(2) != D or o.stride(3) != 1:
                    o = o.contiguous()
                out[s:s + n].copy_(K.merge_extra_key(o, lse.contiguous(), qs, k[extra].unsqueeze(0).contiguous(),
                                                     v[extra].unsqueeze(0).contiguous(), 1 / math.sqrt(D),
                                                     log_mult)[0])
        return out.view(M, C)

    def _block(self, x, a1, ln_r, r, b, mod, segs, nxt):
        """One transformer block. x: residual stream [M, C] fp16; a1: its modulated norm1 [M, C] fp16; ln_r: its RePo
        LayerNorm [M, C] fp32. nxt: (repo, block, mod) of the next block or None. Returns x, a1, ln_r for next."""
        K = self.K
        delta = self._delta(r, ln_r)
        qkv = F.linear(a1, b['qkv_w'], b['qkv_b'])
        o = self._attention(qkv, delta, r, b, segs)
        h = F.linear(o, b['out_w'], b['out_b'])
        if b['modulation']:
            x, a2 = K.add_ln_mod(x, h=h, gate=mod[2], scale=mod[4], shift=mod[3])
        else:
            x, a2 = K.add_ln_mod(x, h=h, weight=b['n2w'], bias=b['n2b'])
        f = K.linear_gelu(a2, b['fc1_w'], b['fc1_b'])
        h = F.linear(f, b['fc2_w'], b['fc2_b'])
        gate = mod[5] if b['modulation'] else None
        if nxt is None:
            x = K.add_ln_mod(x, h=h, gate=gate)[0] if gate is not None else x + h
            return x, None, None
        r2, b2, mod2 = nxt
        if b2['modulation']:
            x, a1, ln_r = K.add_ln_mod(x, h=h, gate=gate, scale=mod2[1], shift=mod2[0], w2=r2['w'], b2=r2['b'])
        else:
            x, a1, ln_r = K.add_ln_mod(x, h=h, gate=gate, weight=b2['n1w'], bias=b2['n1b'], w2=r2['w'], b2=r2['b'])
        return x, a1, ln_r

    def _first_norms(self, x, r, b, mod):
        K = self.K
        if b['modulation']:
            return K.add_ln_mod(x, scale=mod[1], shift=mod[0], w2=r['w'], b2=r['b'])
        return K.add_ln_mod(x, weight=b['n1w'], bias=b['n1b'], w2=r['w'], b2=r['b'])

    def _stack(self, x, layers, mods, segs):
        a1, ln_r = self._first_norms(x, layers[0][0], layers[0][1], mods[0])
        for i, (r, b) in enumerate(layers):
            nxt = (layers[i + 1][0], layers[i + 1][1], mods[i + 1]) if i + 1 < len(layers) else None
            x, a1, ln_r = self._block(x, a1, ln_r, r, b, mods[i], segs, nxt)
        return x

    @staticmethod
    def _mods(t_mod, layers):
        out = []
        for _, b in layers:
            if b['modulation']:
                m = (t_mod + b['shift_table'].type(t_mod.dtype))
                out.append(m.chunk(6, dim=1))
            else:
                out.append(None)
        return out

    # ------------------------------------------------------------------------------------------ public
    @torch.no_grad()
    def prepare(self, cond):
        """Context stream for cond (4,101 tokens) and uncond (zero features: one token if collapse_uncond)."""
        den = self.den
        f1, f2 = cond['feature1'], cond['feature2']
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            h_c = (den.cond_embedder(f1) + den.cond_embedder2(f2)).type(torch.float16)[0].contiguous()
            n_u = 1 if self.collapse else f1.shape[1]
            h_u = (den.cond_embedder(torch.zeros_like(f1[:, :n_u])) +
                   den.cond_embedder2(torch.zeros_like(f2[:, :n_u]))).type(torch.float16)[0].contiguous()
            mods = [None] * len(self.ctx)
            self.h_cond = self._stack(h_c, self.ctx, mods, [(0, h_c.shape[0], None, 0.0)])
            self.h_uncond = self._stack(h_u, self.ctx, mods, [(0, h_u.shape[0], None, 0.0)])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        self.Lc = self.h_cond.shape[0]
        self._lc_full = f1.shape[1]
        self.graph = None
        return self

    @torch.no_grad()
    def velocity(self, z, cam, t, cfg):
        """Guided velocity (1 + w) v(cond) - w v(uncond) for latent z [1, 8192, 16], camera [1, 1, 5], t [1]."""
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            return self._velocity(z, cam, t, cfg)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def _velocity(self, z, cam, t, cfg, return_both=False):
        den, K = self.den, self.K
        Lz, Lc = self.q_len, self.Lc
        t_emb = self._t_embed(t)
        t_mod = den.adaLN_modulation(t_emb).type(torch.float16)
        h_x = den.input_layer(z).type(torch.float16)[0] + self.pos_embed
        mods_n = self._mods(t_mod, self.noise)
        h_x = self._stack(h_x, self.noise, mods_n, [(0, Lz, None, 0.0)])
        h_cam = den.cam_refiner(cam).type(torch.float16)[0]
        Lu = self.h_uncond.shape[0]
        n1 = Lz + Lc + 1
        if self.collapse:
            # uncond rows: latent, camera, context token (its key counts Lc_full times)
            x = torch.cat([h_x, self.h_cond, h_cam, h_x, h_cam, self.h_uncond], 0)
            segs = [(0, n1, None, 0.0), (n1, Lz + 2, n1 + Lz + 1, math.log(self.Lc_full))]
            cam_u = n1 + Lz
        else:
            x = torch.cat([h_x, self.h_cond, h_cam, h_x, self.h_uncond, h_cam], 0)
            segs = [(0, n1, None, 0.0), (n1, Lz + Lu + 1, None, 0.0)]
            cam_u = n1 + Lz + Lu
        mods = self._mods(t_mod, self.joint)
        x = self._stack(x, self.joint, mods, segs)
        rows = torch.cat([x[0:Lz], x[n1:n1 + Lz]], 0).float()
        cams = torch.stack([x[n1 - 1], x[cam_u]], 0).float()
        rows = F.layer_norm(rows, (C,))
        cams = F.layer_norm(cams, (C,))
        shift, scale = (den.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        rows = rows * (1 + scale[0]) + shift[0]
        cams = cams * (1 + scale[0]) + shift[0]
        vz = den.out_layer(rows)
        vc = den.cam_out_layer(cams)
        vz_c, vz_u = vz[:Lz].unsqueeze(0), vz[Lz:].unsqueeze(0)
        vc_c, vc_u = vc[0:1].unsqueeze(0), vc[1:2].unsqueeze(0)
        if return_both:
            return (vz_c, vc_c), (vz_u, vc_u)
        return (1 + cfg) * vz_c - cfg * vz_u, (1 + cfg) * vc_c - cfg * vc_u

    @torch.no_grad()
    def both(self, z, cam, t):
        """(v_cond, v_uncond) for teacher-forced comparisons."""
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            return self._velocity(z, cam, t, 0.0, return_both=True)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    @property
    def Lc_full(self):
        return self._lc_full

    @torch.no_grad()
    def sample(self, noise, cond, steps=50, rescale_t=3.0, cfg=3.0):
        """sample_flow with the reference's time grid (numpy float64) and Euler update."""
        import numpy as np
        self.prepare(cond)
        z, cam = noise['latent'], noise['camera']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        if self.cuda_graph:
            return self._sample_graph(z, cam, t_seq, cfg)
        for i in range(steps):
            t, t_prev = t_seq[i], t_seq[i + 1]
            tt = torch.tensor([1000 * t], device=z.device, dtype=torch.float32)
            vz, vc = self.velocity(z, cam, tt, cfg)
            z = z - vz * (t - t_prev)
            cam = cam - vc * (t - t_prev)
        return z, cam

    def _sample_graph(self, z, cam, t_seq, cfg):
        """One CUDA graph for a whole step (velocity + Euler update), replayed per step with t and dt in device
        buffers."""
        steps = len(t_seq) - 1
        if self.graph is None:
            self.g_z = z.clone()
            self.g_cam = cam.clone()
            self.g_t = torch.zeros(1, device=z.device)
            self.g_dt = torch.zeros(1, device=z.device)
            self.g_t.fill_(1000 * t_seq[0])
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):  # warm up (cuDNN plans, cuBLAS workspaces) outside the capture
                    self.velocity(self.g_z, self.g_cam, self.g_t, cfg)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                vz, vc = self.velocity(self.g_z, self.g_cam, self.g_t, cfg)
                self.g_z.sub_(vz * self.g_dt)
                self.g_cam.sub_(vc * self.g_dt)
            self.graph = g
        self.g_z.copy_(z)
        self.g_cam.copy_(cam)
        for i in range(steps):
            # same arithmetic as the eager loop: t in fp32, dt = (t - t_prev) as a float64 scalar rounded to fp32
            self.g_t.fill_(1000 * t_seq[i])
            self.g_dt.fill_(t_seq[i] - t_seq[i + 1])
            self.graph.replay()
        return self.g_z.clone(), self.g_cam.clone()
