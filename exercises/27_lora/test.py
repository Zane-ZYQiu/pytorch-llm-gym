import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_shape():
    m = task.LoRALinear(8, 5, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert m(x).shape == (3, 5)


def test_identity_at_init():
    # when B=0, the output should equal the pure base x @ W^T
    m = task.LoRALinear(8, 5, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    base = x @ m.weight.t()
    assert torch.allclose(m(x), base, atol=1e-6), "initial (B=0) output should equal the frozen base"


def test_value_with_nonzero_B():
    m = task.LoRALinear(8, 5, r=2, alpha=4.0)
    with torch.no_grad():
        m.B.copy_(torch.randn_like(m.B))
    x = torch.randn(3, 8)
    expected = x @ m.weight.t() + (x @ m.A.t()) @ m.B.t() * m.scaling
    assert torch.allclose(m(x), expected, atol=1e-5)


def test_only_AB_trainable():
    m = task.LoRALinear(8, 5, r=2, alpha=4.0)
    assert m.weight.requires_grad is False, "base weight should be frozen"
    assert m.A.requires_grad and m.B.requires_grad
    # after backward, only A and B have gradients
    x = torch.randn(3, 8)
    with torch.no_grad():
        m.B.copy_(torch.randn_like(m.B))
    m(x).sum().backward()
    assert m.weight.grad is None
    assert m.A.grad is not None and m.B.grad is not None
