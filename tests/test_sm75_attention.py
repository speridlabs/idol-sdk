import unittest

import torch

from idol.models.layers import scaled_dot_product_attention
from idol.models.sm75_attention import same_queries


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5),
                     "requires a Turing CUDA device")
class SameQueriesTest(unittest.TestCase):
    def test_compiled_attention_matches_sdpa(self):
        q, k, v = torch.randn(2, 12294, 3, 16, 64, device='cuda', dtype=torch.float16).unbind(2)
        compiled = torch.compile(same_queries, fullgraph=True)
        for constant in (False, True):
            with self.subTest(constant=constant):
                if constant:
                    q[1, 8192:-1] = q[1, 8192:8193].clone()
                torch.testing.assert_close(compiled(q, k, v), scaled_dot_product_attention(q, k, v),
                                           atol=0, rtol=0)
