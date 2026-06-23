import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def _reference(x, router, experts, k):
    N, D = x.shape
    gate = x @ router
    out = torch.zeros_like(x)
    for n in range(N):
        vals, idx = torch.topk(gate[n], k)
        w = torch.softmax(vals, dim=-1)
        for j in range(k):
            e = idx[j].item()
            out[n] = out[n] + w[j] * (x[n] @ experts[e])
    return out


def _make(N, D, E):
    torch.manual_seed(0)
    return (torch.randn(N, D), torch.randn(D, E), torch.randn(E, D, D))


def test_shape():
    x, router, experts = _make(5, 8, 4)
    out = task.moe_layer(x, router, experts, top_k=2)
    assert out.shape == (5, 8)


def test_matches_reference_top2():
    x, router, experts = _make(6, 8, 4)
    out = task.moe_layer(x, router, experts, top_k=2)
    assert torch.allclose(out, _reference(x, router, experts, 2), atol=1e-4)


def test_top1_routing():
    x, router, experts = _make(6, 8, 5)
    out = task.moe_layer(x, router, experts, top_k=1)
    # with top_k=1 the softmax weight is always 1, so the output is the chosen expert's raw output
    assert torch.allclose(out, _reference(x, router, experts, 1), atol=1e-4)


def test_all_experts():
    x, router, experts = _make(4, 8, 3)
    out = task.moe_layer(x, router, experts, top_k=3)
    assert torch.allclose(out, _reference(x, router, experts, 3), atol=1e-4)
