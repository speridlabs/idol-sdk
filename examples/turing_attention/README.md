# Experimental full attention on Turing

This opt-in example uses the standard IDOL CLI and replaces denoiser self-attention with the noncausal FP16 forward kernel from [FlashAttention Turing](https://github.com/ssiu/flash-attention-turing/tree/9ef98fcb506bb1e2fe3cece50935e2935bf6b124). It computes all query/key interactions on every sampling step. Accumulation order differs from PyTorch SDPA, so the two backends can produce different PLYs.

Build from the repository root with an SM75-capable CUDA compiler and the normal SDK environment. The build fetches a pinned upstream revision and its CUTLASS submodule into a temporary checkout. It compiles only the 64-wide noncausal forward specialization. Use a dedicated SDK checkout for the benchmark. On the RTX 2080, we used FP16 for the DINO and VAE encoders. Apply the included encoder adjustment once to that checkout so the candidate and control use the same encoder precision. Set CUDA_VISIBLE_DEVICES to the Turing GPU index or UUID reported by nvidia-smi.

```bash
git apply examples/turing_attention/encoder-fp16.patch
export ATTN_BACKEND=sdpa
examples/turing_attention/build.sh
PYTHONPATH=. python examples/turing_attention/sample.py --device cuda --mode synthetic \
  --image object.png --steps 50 --num_gaussians 1048576 --out native-output
PYTHONPATH=. python -m idol.sample --device cuda --mode synthetic \
  --image object.png --steps 50 --num_gaussians 1048576 --out control-output
```

Use the Python SDK's standard attention backend for the control (ATTN_BACKEND=sdpa). The example expects matching FP16 query/key/value shapes on SM75 and the default CUDA stream. Decoding and the regular SDK entry point retain their original attention code. This is an experimental launcher for benchmarking, not a mandatory installation dependency.

On one RTX 2080, an alternating test of warmed denoiser forwards on identical inputs measured 1.976 seconds with SDK SDPA and 1.623 seconds with the Turing backend (four samples of two forwards per backend). In two 50-step runs in the downstream application, denoising fell from 107.1 to 93.7 seconds on the first image and from 102.8 to 80.8 seconds on the second. A concurrent Blender process briefly shared the device during the control run. Fixed-camera silhouette IoU averaged 0.9972 and 0.9966 across nine views for those two subjects. These are preliminary measurements on one card.
