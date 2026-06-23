import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_mean_and_var():
    x = torch.randn(4, 6)
    mean, var = task.mean_and_var(x, dim=1)
    assert mean.shape == (4, 1) and var.shape == (4, 1), "should be keepdim=True"
    assert torch.allclose(mean, x.mean(dim=1, keepdim=True))
    assert torch.allclose(var, x.var(dim=1, unbiased=False, keepdim=True), atol=1e-5)


def test_standardize_stats():
    x = torch.randn(3, 10) * 4 + 7
    out = task.standardize(x, dim=1)
    assert out.shape == x.shape
    # after standardization, mean ≈ 0 and variance ≈ 1
    assert torch.allclose(out.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.var(dim=1, unbiased=False), torch.ones(3), atol=1e-3)


def test_standardize_formula():
    x = torch.randn(2, 5)
    out = task.standardize(x, dim=1, eps=1e-5)
    m = x.mean(dim=1, keepdim=True)
    v = x.var(dim=1, unbiased=False, keepdim=True)
    expected = (x - m) / torch.sqrt(v + 1e-5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_global_grad_norm():
    g1 = torch.tensor([3.0, 4.0])      # norm 5
    g2 = torch.tensor([[0.0, 0.0]])    # norm 0
    out = task.global_grad_norm([g1, g2])
    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(5.0))

    ts = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 2)]
    flat = torch.cat([t.flatten() for t in ts])
    assert torch.allclose(task.global_grad_norm(ts), flat.norm(), atol=1e-4)
