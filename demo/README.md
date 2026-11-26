---
title: Idol
emoji: 🗿
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.49.1
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: One image to a 3D object (Gaussian splats)
---

# Idol demo

Speridlabs Gradio demo. `app.py` is the UI; `model.py` is the model.
`examples/presets.json` lists each example image with the settings it is run with.

The Space needs an `HF_TOKEN` secret from an account with access to
`facebook/dinov3-vith16plus-pretrain-lvd1689m` and `black-forest-labs/FLUX.2-dev` (the image encoders).

On ZeroGPU, requests run the fast build from [idol-cuda](https://github.com/suvadityamuk/idol-cuda) (Apache-2.0,
commit 3fa7ca6): `idol.fast`, with its kernels pinned to one commit of the Hub kernel repo `suvadityamuk/idol-cuda`.
It reuses idol's encoders and decoders and replaces the sampler, about 1.8x faster than the eager pipeline there.
