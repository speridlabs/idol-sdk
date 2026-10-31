# Idol

**A single image to 3DGS object model**

Idol turns one picture of an object into a 3D Gaussian splat. Give it a render, a generated image or a photo: it removes the background, frames the object and generates up to 1,048,576 Gaussians, including the sides the picture never showed. It runs on consumer GPUs, using 7.2 GB of VRAM and can run in under 2 seconds.

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/speridlabs/idol)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/speridlabs/idol)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

| ![](assets/gallery/armchair.webp) | ![](assets/gallery/avocado.webp) |
|---|---|
| ![](assets/gallery/lamp.webp) | ![](assets/gallery/axolotl.webp) |

Each pair: the input image and the generated object turning. The
[demo](https://huggingface.co/spaces/speridlabs/idol) has 93 more examples ready to run and takes your own images as well.

## Model

| | |
|---|---|
| Output | 3D Gaussians: 32,768 to 1,048,576 (default 262,144), standard 3DGS `.ply` |
| DiT | 370M parameters |
| Image conditioning | DINOv3 ViT-H+ and FLUX.2 VAE |
| Decoder | 288M parameters |
| Background removal | BiRefNet |
| Sampler | Euler, 50 steps, classifier-free guidance 3.0, time shift 3.0 |

## Install

```bash
pip install -e .              # Python >= 3.10, PyTorch >= 2.5
pip install -e ".[demo]"      # + Gradio, to run the demo locally
```

The weights are downloaded from Hugging Face on first use, you need to request access to
[facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
and [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) first, then log in with
`hf auth login`. 

`flash-attn`, if installed, is used for attention; otherwise PyTorch's `scaled_dot_product_attention` is (same
result, slower). Set `ATTN_BACKEND=sdpa` or `ATTN_BACKEND=flash_attn` to choose.

## Inference

```bash
python -m idol.sample --image photo.jpg --mode real
python -m idol.sample --image renders/ --mode synthetic --out outputs
```

For each image this writes `<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the image the model saw, after background
removal and cropping) and `<name>_camera.json` (the predicted camera of the input).

**Pick the mode for your input:** 

| `--mode` | for | 
|---|---|
| `synthetic` (default) | computer generated images| 
| `real` | real world photographs | 

Other flags:

| flag | default | |
|---|---|---|
| `--seed` | 42 | same seed, same object |
| `--steps` | 50 | sampling steps |
| `--cfg_strength` | 3.0 | classifier-free guidance |
| `--rescale_t` | 3.0 | time shift: higher spends more steps at high noise |
| `--num_gaussians` | 262144 | 32,768 to 1,048,576, a multiple of 32 |
| `--dinov3_dtype` | bf16 | `fp32` runs the DINOv3 encoder in full precision (1.7 GB more memory) |
| `--weights` | `speridlabs/idol` | a Hugging Face repo or a local folder |
| `--no_optimize` | optimized | turn off the faster sampling (see below) |

**Faster sampling is on by default on GPU** The first image in a process is slower while it pytorch compiles; the rest run at full
speed. 

### Python API

```python
from idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
output = pipeline("photo.jpg", mode="real", seed=42, num_gaussians=1_048_576)
output.gaussian.save_ply("photo.ply")
output.preprocessed_image.save("photo_input.png")
```

The image encoders come from the Hugging Face repos named in the weights' `config.json`. `image_encoders` takes them
from somewhere else, by name: another repo id, a local folder in the same layout (FLUX.2's with its autoencoder under
`vae/`), or a model you have already loaded. Only repo ids are checked for gated access, so local copies load offline
without a token:

```python
pipeline = IdolPipeline.from_pretrained("speridlabs/idol", image_encoders={
    "dinov3": "/models/dinov3-vith16plus-pretrain-lvd1689m",  # or a loaded transformers DINOv3 model
    "flux2_vae": "/models/FLUX.2-klein-4B",                   # any FLUX.2 repo with the same autoencoder in vae/
})
```

### Demo

The Gradio app behind the [Hugging Face Space](https://huggingface.co/spaces/speridlabs/idol) is in `demo/`,
with the example images. It runs on ZeroGPU there and on any local GPU:

```bash
pip install -e ".[demo]"
cd demo && python app.py
```

## More details about the model

| Component | Parameters | Precision on GPU | Weights in VRAM |
|---|---|---|---|
| DiT (denoiser) | 370.6M | FP16 transformer blocks; FP32 embeddings and norms | ~0.8–1.0 GB |
| Octree decoder | 50.8M | FP16 transformer blocks; FP32 embeddings and norms | ~0.1–0.2 GB |
| Gaussian decoder | 237.3M | FP16 transformer blocks; FP32 embeddings and norms | ~0.5–0.9 GB |
| BiRefNet (background removal) | 220.7M | FP16 | ~0.44 GB |
| DINOv3 ViT-H+/16  | 840.6M | bf16 | ~1.68 GB |
| FLUX.2 VAE  | 84.1M  | bf16 (FP32 checkpoint, cast on load) | ~0.17 GB |
| **Full request** | **~1.80B** | mostly FP16 / bf16, some FP32 | **~3.7–4.4 GB of weights, 7.2 GB peak** |

## Benchmark

Idol, TripoSplat, TRELLIS and LGM, scored at the input view (PSNR, SSIM, LPIPS, IoU) and at seven new views
(DINOv2, CLIP, sharpness distance). Mean scores, 1 × H100 per model. Bold is the best in each column.

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | DINOv2 ↑ | CLIP ↑ | Sharpness distance ↓ | Request time ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Idol** | 50 steps, 262,144 Gaussians | 12.25 | **0.726** | **0.213** | **0.862** | **0.781** | **0.885** | **0.81** | 6.8 s |
| TripoSplat | 50 steps, 262,144 Gaussians | 11.76 | 0.712 | 0.225 | 0.860 | 0.743 | 0.868 | 1.03 | 9.9 s |
| TRELLIS | Paper setting, 50 + 50 steps | 11.89 | 0.716 | 0.235 | 0.845 | 0.754 | 0.868 | 0.97 | 7.1 s |
| LGM | Default, 65,536 Gaussians | **12.46**\* | 0.722 | 0.237 | 0.834 | 0.575 | 0.784 | 1.22 | **1.1 s** |

\* LGM has the highest PSNR despite visibly blurry results, because PSNR averages squared pixel error.

## Across GPUs

One image to 262,144 Gaussians in 3.4 s. Request time at 50 steps: median over 8 images, with the same code and
settings on every GPU, one GPU per machine.

| GPU | VRAM | Request time |
|---|---|---|
| B300 | 269 GB | 3.4 s |
| B200 | 179 GB | 3.6 s |
| H200 | 140 GB | 6.6 s |
| H100 | 80 GB | 6.7 s |
| RTX 5090 | 32 GB | 14.1 s |
| A100 | 80 GB | 16.3 s |
| L40S | 45 GB | 16.6 s |
| RTX 4090 | 24 GB | 19.1 s |
| RTX 3080 | 10 GB | 46.2 s |
| L4 | 22 GB | 54.6 s |

## Acknowledgements

Idol builds on [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat). It uses
[DINOv3](https://github.com/facebookresearch/dinov3), the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev)
VAE and [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

**Made with ❤️ by [speridlabs.com](https://speridlabs.com)**
