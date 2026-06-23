import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_cross_entropy_value():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    out = task.cross_entropy(logits, targets)
    assert out.shape == ()
    assert torch.allclose(out, F.cross_entropy(logits, targets), atol=1e-5)


def test_cross_entropy_stable():
    logits = torch.tensor([[0.0, 1000.0, -1000.0]])
    targets = torch.tensor([1])
    out = task.cross_entropy(logits, targets)
    assert torch.isfinite(out).all()


def test_cross_entropy_grad():
    logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    g = task.cross_entropy_grad(logits.detach(), targets)
    assert g.shape == (8, 10)
    assert torch.allclose(g, logits.grad, atol=1e-5), \
        "gradient should equal (softmax - onehot)/N, matching autograd"


def test_cross_entropy_grad_is_softmax_minus_onehot():
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 2, 4, 1])
    g = task.cross_entropy_grad(logits, targets)
    p = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(targets, 5).float()
    assert torch.allclose(g, (p - onehot) / 4, atol=1e-6)
