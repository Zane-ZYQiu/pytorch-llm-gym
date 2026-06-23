import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_linear_forward():
    x = torch.randn(4, 5)
    W = torch.randn(3, 5)
    b = torch.randn(3)
    out = task.linear_forward(x, W, b)
    assert out.shape == (4, 3)
    assert torch.allclose(out, F.linear(x, W, b), atol=1e-5)


def test_linear_backward_matches_autograd():
    x = torch.randn(4, 5, requires_grad=True)
    W = torch.randn(3, 5, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    grad_out = torch.randn(4, 3)

    y = F.linear(x, W, b)
    y.backward(grad_out)  # use grad_out as the upstream gradient

    gx, gW, gb = task.linear_backward(x.detach(), W.detach(), grad_out)
    assert gx.shape == (4, 5) and gW.shape == (3, 5) and gb.shape == (3,)
    assert torch.allclose(gx, x.grad, atol=1e-5), "grad_x is wrong"
    assert torch.allclose(gW, W.grad, atol=1e-5), "grad_W is wrong"
    assert torch.allclose(gb, b.grad, atol=1e-5), "grad_b is wrong"
