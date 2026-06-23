import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def _reference(x, Wq, Wk, Wv, Wo, H, causal):
    B, T, D = x.shape
    hd = D // H
    q = (x @ Wq).reshape(B, T, H, hd).transpose(1, 2)
    k = (x @ Wk).reshape(B, T, H, hd).transpose(1, 2)
    v = (x @ Wv).reshape(B, T, H, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # (B,H,T,hd)
    o = o.transpose(1, 2).reshape(B, T, D)
    return o @ Wo


def test_mha_shape():
    x = torch.randn(2, 5, 12)
    Wq, Wk, Wv, Wo = [torch.randn(12, 12) for _ in range(4)]
    out = task.multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads=3)
    assert out.shape == (2, 5, 12)


def test_mha_matches_reference():
    x = torch.randn(2, 6, 16)
    Wq, Wk, Wv, Wo = [torch.randn(16, 16) for _ in range(4)]
    out = task.multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads=4)
    ref = _reference(x, Wq, Wk, Wv, Wo, 4, False)
    assert torch.allclose(out, ref, atol=1e-4)


def test_mha_causal():
    x = torch.randn(2, 6, 16)
    Wq, Wk, Wv, Wo = [torch.randn(16, 16) for _ in range(4)]
    out = task.multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads=4, causal=True)
    ref = _reference(x, Wq, Wk, Wv, Wo, 4, True)
    assert torch.allclose(out, ref, atol=1e-4)
    # causality: changing the last token must not affect the outputs of earlier tokens
    x2 = x.clone()
    x2[:, -1] += 5.0
    out2 = task.multi_head_attention(x2, Wq, Wk, Wv, Wo, num_heads=4, causal=True)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-4)
