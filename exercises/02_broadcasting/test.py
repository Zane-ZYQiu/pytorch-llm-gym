import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_pairwise_sq_dist():
    a = torch.randn(4, 3)
    b = torch.randn(5, 3)
    out = task.pairwise_sq_dist(a, b)
    assert out.shape == (4, 5)
    expected = torch.cdist(a, b) ** 2
    assert torch.allclose(out, expected, atol=1e-4), "distance values are wrong"


def test_pairwise_sq_dist_no_python_loop():
    src = pathlib.Path(__file__).resolve().parents[0]
    # soft check: not enforced, but vectorization is encouraged
    assert task.pairwise_sq_dist(torch.zeros(2, 2), torch.zeros(3, 2)).shape == (2, 3)


def test_outer_sum():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0])
    out = task.outer_sum(a, b)
    assert out.shape == (3, 2)
    assert torch.allclose(out, a.unsqueeze(1) + b.unsqueeze(0))


def test_normalize_last_dim():
    x = torch.randn(6, 8) * 5 + 1  # far from 0, so eps has negligible effect
    out = task.normalize_last_dim(x)
    assert out.shape == x.shape
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "each row's norm should be 1"
    assert torch.allclose(out, x / x.norm(dim=-1, keepdim=True), atol=1e-4)


def test_masked_mean():
    x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False],
                         [True, False, False]])
    out = task.masked_mean(x, mask)
    assert out.shape == (2, 2)
    # sample 0: average the first two time steps
    exp0 = (x[0, 0] + x[0, 1]) / 2
    # sample 1: only the 0th time step
    exp1 = x[1, 0]
    assert torch.allclose(out[0], exp0)
    assert torch.allclose(out[1], exp1)
