"""Device selection for inference.

Idol runs on CUDA, Apple Silicon MPS and CPU. The pipeline, CLI and demo all call ``resolve_device`` so
they agree on the same order: an explicit device, then ``IDOL_DEVICE``, then CUDA, MPS, CPU.
"""
import os

import torch


def resolve_device(device=None) -> torch.device:
    """Return ``device`` when given, else ``IDOL_DEVICE``, else the best available: CUDA, MPS, CPU."""
    if device is not None:
        return torch.device(device)
    requested = os.environ.get('IDOL_DEVICE')
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
