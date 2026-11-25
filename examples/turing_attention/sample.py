"""Run the standard sampling CLI with experimental SM75 denoiser attention."""
import ctypes
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from idol.models import layers
from idol.pipeline import IdolPipeline
from idol.sample import main

HERE = Path(__file__).resolve().parent


def run_candidate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This benchmark kernel was built for SM75")
    library = ctypes.CDLL(str(HERE / "libfwd.so"))
    library.run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3

    @torch.library.custom_op("idolprobe::turing_attention", mutates_args=())
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if (q.ndim != 4 or q.shape[-1] != 64 or k.shape != q.shape or v.shape != q.shape
                or any(t.dtype != torch.float16 or not t.is_cuda or t.device != q.device for t in (q, k, v))):
            raise ValueError("Expected matching CUDA FP16 [batch, length, heads, 64] tensors")
        if q.device.index != torch.cuda.current_device() or torch.cuda.current_stream().cuda_stream != 0:
            raise RuntimeError("The benchmark launcher requires the current GPU and its default stream")
        q, k, v = (t.contiguous() for t in (q, k, v))
        batch, length, heads, _ = q.shape
        output = torch.empty_like(q)
        lse = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
        status = library.run(q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(),
                             lse.data_ptr(), batch, length, heads)
        if status:
            raise RuntimeError(f"Turing CUDA kernel launch failed with status {status}")
        return output

    @attention.register_fake
    def fake(q, k, v):
        return torch.empty_like(q, memory_format=torch.contiguous_format)

    original_settings = IdolPipeline._sampling_settings

    @contextmanager
    def settings(pipeline):
        with original_settings(pipeline), patch.object(layers, "scaled_dot_product_attention", attention):
            yield

    with patch.object(IdolPipeline, "_sampling_settings", settings):
        main()


if __name__ == "__main__":
    run_candidate()
