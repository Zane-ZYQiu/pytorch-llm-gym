import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_shapes_first_and_later_step():
    B, H, hd = 2, 3, 4
    q = torch.randn(B, H, 1, hd)
    k = torch.randn(B, H, 1, hd)
    v = torch.randn(B, H, 1, hd)
    out, kc, vc = task.append_and_attend(q, k, v, None, None)
    assert out.shape == (B, H, 1, hd)
    assert kc.shape == (B, H, 1, hd) and vc.shape == (B, H, 1, hd)
    # second step
    out2, kc2, vc2 = task.append_and_attend(q, k, v, kc, vc)
    assert kc2.shape == (B, H, 2, hd)


def test_incremental_equals_full_causal():
    """Core: decoding step by step with a KV cache should equal one-shot causal attention."""
    B, H, T, hd = 2, 3, 7, 8
    q = torch.randn(B, H, T, hd)
    k = torch.randn(B, H, T, hd)
    v = torch.randn(B, H, T, hd)

    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,H,T,hd)

    kc, vc = None, None
    outs = []
    for t in range(T):
        out_t, kc, vc = task.append_and_attend(
            q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], kc, vc)
        outs.append(out_t)
    incremental = torch.cat(outs, dim=2)  # (B,H,T,hd)

    assert torch.allclose(incremental, full, atol=1e-4), \
        "incremental decoding result should be exactly identical to one-shot causal attention"
