import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_softmax_matches_torch():
    x = torch.randn(3, 6)
    out = task.softmax(x, dim=-1)
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(out.sum(-1), torch.ones(3), atol=1e-6), "each row should sum to 1"


def test_softmax_other_dim():
    x = torch.randn(4, 5)
    out = task.softmax(x, dim=0)
    assert torch.allclose(out, torch.softmax(x, dim=0), atol=1e-6)


def test_softmax_numerically_stable():
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    out = task.softmax(x, dim=-1)
    assert torch.isfinite(out).all(), "should not produce nan/inf for large values"
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)


def test_logsumexp():
    x = torch.randn(3, 7)
    out = task.logsumexp(x, dim=-1)
    assert out.shape == (3,), "logsumexp should remove the dim axis"
    assert torch.allclose(out, torch.logsumexp(x, dim=-1), atol=1e-5)
    # stability
    big = torch.tensor([1000.0, 1000.0])
    assert torch.isfinite(task.logsumexp(big, dim=-1)).all()


def test_log_softmax():
    x = torch.randn(3, 6)
    out = task.log_softmax(x, dim=-1)
    assert torch.allclose(out, torch.log_softmax(x, dim=-1), atol=1e-5)
    big = torch.tensor([[1000.0, 1001.0, 1002.0]])
    assert torch.isfinite(task.log_softmax(big, dim=-1)).all()
