import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_matches_full_attention_various_blocks():
    B, H, T, d = 2, 3, 10, 8
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)
    full = F.scaled_dot_product_attention(q, k, v)  # non-causal
    for bs in [1, 2, 3, 5, 10, 16]:
        out = task.flash_attention(q, k, v, block_size=bs)
        assert out.shape == (B, H, T, d)
        assert torch.allclose(out, full, atol=1e-4), f"with block_size={bs} the result should match standard attention"


def test_numerically_stable_large_scores():
    # no nan/inf should appear for large values (online softmax subtracts the max)
    q = torch.randn(1, 1, 6, 4) * 50
    k = torch.randn(1, 1, 6, 4) * 50
    v = torch.randn(1, 1, 6, 4)
    out = task.flash_attention(q, k, v, block_size=2)
    assert torch.isfinite(out).all()
