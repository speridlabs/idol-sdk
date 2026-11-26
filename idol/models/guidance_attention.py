"""Full SDPA for repeated queries in the unconditional guidance branch."""
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext


@torch.library.custom_op("idol::same_queries", mutates_args=())
def same_queries(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cudnn: bool) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if cudnn else nullcontext():
        return _same_queries(q, k, v)


def _same_queries(q, k, v):
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    # The joint sequence is [8192 object tokens, image tokens, one camera].
    # Only the second classifier-free guidance branch has a constant input image.
    start = 8192
    image_queries = q[1:, :, start:-1]
    representative = image_queries[:, :, :1]
    if not torch.equal(image_queries, representative.expand_as(image_queries)):
        return scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()

    positive = scaled_dot_product_attention(q[:1], k[:1], v[:1])
    compact_q = torch.cat((q[1:, :, :start + 1], q[1:, :, -1:]), dim=2)
    # Keep every key and value, including their multiplicity in the softmax.
    compact = scaled_dot_product_attention(compact_q, k[1:], v[1:])
    image_output = compact[:, :, start:start + 1].expand_as(image_queries)
    negative = torch.cat((compact[:, :, :start], image_output, compact[:, :, -1:]), dim=2)
    return torch.cat((positive, negative)).transpose(1, 2).contiguous()


@same_queries.register_fake
def _fake(q, k, v, cudnn):
    return torch.empty_like(q, memory_format=torch.contiguous_format)
