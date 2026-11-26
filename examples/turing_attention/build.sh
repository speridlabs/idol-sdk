#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git clone --quiet https://github.com/ssiu/flash-attention-turing "$work/flash"
git -C "$work/flash" checkout --quiet 9ef98fcb506bb1e2fe3cece50935e2935bf6b124
git -C "$work/flash" submodule update --init --depth 1
src="$work/flash/csrc"
# This forward header uses no PyTorch types. Compile just the CUDA implementation.
sed -i '/^#include <torch\/extension.h>/d' "$src/flash_attn/src/flash_fwd_kernel.h"
nvcc -std=c++17 -allow-unsupported-compiler --expt-relaxed-constexpr -arch=sm_75 \
    -O3 --use_fast_math -Xcompiler -fPIC -shared -I"$src/flash_attn/src" \
    -I"$src/cutlass/include" "$here/forward.cu" -o "$here/libfwd.so"
