import os

import torch


def resolve_device(device=None) -> torch.device:
    """The given device, else IDOL_DEVICE, else CUDA, then MPS, then CPU."""
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
