"""Loads the idol_cuda kernels from the Hub, pinned to one commit (replaces idol-cuda's fast/ops.py)."""
from kernels import get_kernel

REPO = 'suvadityamuk/idol-cuda'
REVISION = 'b82869a5a22361373dd8d1ae25a6a3cbed848316'  # the repo's v1 branch on 2026-11-26
_K = None


def kernel():
    global _K
    if _K is None:
        _K = get_kernel(REPO, revision=REVISION, trust_remote_code=[REPO])
    return _K
