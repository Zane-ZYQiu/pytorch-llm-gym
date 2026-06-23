import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_grad_of_quadratic():
    x = torch.tensor([1.0, -2.0, 3.0])
    g = task.grad_of(lambda t: (t ** 2).sum(), x)
    # d/dx sum(x^2) = 2x
    assert torch.allclose(g, 2 * x), f"expected {2*x}, got {g}"


def test_grad_of_does_not_mutate_input():
    x = torch.tensor([1.0, 2.0])
    assert x.requires_grad is False
    _ = task.grad_of(lambda t: (t ** 2).sum(), x)
    assert x.requires_grad is False, "should not modify the passed-in x"
    assert x.grad is None


def test_numerical_grad_matches_analytic():
    # gradient checks conventionally use float64 to avoid cancellation noise from subtracting differences in float32
    x = torch.randn(5, dtype=torch.float64)
    f = lambda t: torch.sin(t).sum()
    num = task.numerical_grad(f, x)
    ana = torch.cos(x)  # d/dx sum(sin x) = cos x
    assert torch.allclose(num, ana, atol=1e-5), "numerical gradient does not match the analytic gradient"


def test_numerical_and_autograd_agree():
    x = torch.randn(4, 3, dtype=torch.float64)
    f = lambda t: (t ** 3).sum()
    num = task.numerical_grad(f, x)
    auto = task.grad_of(f, x)
    assert torch.allclose(num, auto, atol=1e-5), "numerical gradient should be close to the autograd result"
