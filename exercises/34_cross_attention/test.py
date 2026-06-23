import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def _reference(x, context, Wq, Wk, Wv, Wo, H):
    B, Tx, D = x.shape
    Tc = context.shape[1]
    hd = D // H
    q = (x @ Wq).reshape(B, Tx, H, hd).transpose(1, 2)
    k = (context @ Wk).reshape(B, Tc, H, hd).transpose(1, 2)
    v = (context @ Wv).reshape(B, Tc, H, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.transpose(1, 2).reshape(B, Tx, D)
    return o @ Wo


def test_shape():
    x = torch.randn(2, 5, 16)        # 5 text tokens
    ctx = torch.randn(2, 9, 16)      # 9 image patches
    Ws = [torch.randn(16, 16) for _ in range(4)]
    out = task.cross_attention(x, ctx, *Ws, num_heads=4)
    assert out.shape == (2, 5, 16)


def test_matches_reference():
    x = torch.randn(2, 5, 16)
    ctx = torch.randn(2, 9, 16)
    Wq, Wk, Wv, Wo = [torch.randn(16, 16) for _ in range(4)]
    out = task.cross_attention(x, ctx, Wq, Wk, Wv, Wo, num_heads=4)
    ref = _reference(x, ctx, Wq, Wk, Wv, Wo, 4)
    assert torch.allclose(out, ref, atol=1e-4)


def test_depends_on_context():
    x = torch.randn(2, 5, 16)
    ctx = torch.randn(2, 9, 16)
    Ws = [torch.randn(16, 16) for _ in range(4)]
    out1 = task.cross_attention(x, ctx, *Ws, num_heads=4)
    out2 = task.cross_attention(x, torch.randn(2, 9, 16), *Ws, num_heads=4)
    assert not torch.allclose(out1, out2, atol=1e-3), "changing the context should change the output"
