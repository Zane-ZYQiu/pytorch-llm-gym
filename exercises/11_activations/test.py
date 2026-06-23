import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_silu():
    x = torch.randn(10)
    assert torch.allclose(task.silu(x), F.silu(x), atol=1e-6)


def test_gelu_exact():
    x = torch.randn(10)
    assert torch.allclose(task.gelu(x), F.gelu(x), atol=1e-6)


def test_swiglu_shape_and_value():
    D, F_dim = 8, 20
    x = torch.randn(2, 5, D)
    W_gate = torch.randn(D, F_dim)
    W_up = torch.randn(D, F_dim)
    W_down = torch.randn(F_dim, D)
    out = task.swiglu_ffn(x, W_gate, W_up, W_down)
    assert out.shape == (2, 5, D)
    ref = (F.silu(x @ W_gate) * (x @ W_up)) @ W_down
    assert torch.allclose(out, ref, atol=1e-5)
