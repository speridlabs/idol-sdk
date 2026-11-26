import unittest

import torch

from idol.models.layers import scaled_dot_product_attention
from idol.models.guidance_attention import same_queries


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() in ((7, 5), (12, 0)),
                     "requires SM75 or SM120")
class SameQueriesTest(unittest.TestCase):
    def test_compiled_attention_matches_sdpa(self):
        q, k, v = torch.randn(2, 12294, 3, 16, 64, device='cuda', dtype=torch.float16).unbind(2)
        from unittest.mock import patch
        cudnn = torch.cuda.get_device_capability() == (12, 0)
        compiled = torch.compile(same_queries, fullgraph=True)
        for constant in (False, True):
            with self.subTest(constant=constant), patch("idol.models.layers.ATTN_BACKEND", "sdpa_cudnn" if cudnn else "sdpa"):
                if constant:
                    q[1, 8192:-1] = q[1, 8192:8193].clone()
                torch.testing.assert_close(compiled(q, k, v, cudnn), scaled_dot_product_attention(q, k, v),
                                           atol=0, rtol=0)
