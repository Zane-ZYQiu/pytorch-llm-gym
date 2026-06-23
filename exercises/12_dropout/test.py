import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_eval_is_identity():
    x = torch.randn(100)
    out = task.dropout(x, p=0.5, training=False)
    assert torch.equal(out, x), "should return x unchanged when training=False"


def test_p_zero_is_identity():
    x = torch.randn(100)
    out = task.dropout(x, p=0.0, training=True)
    assert torch.allclose(out, x), "should return x unchanged when p=0"


def test_dropout_statistics():
    torch.manual_seed(0)
    x = torch.ones(50000)
    p = 0.3
    out = task.dropout(x, p=p, training=True)
    zero_frac = (out == 0).float().mean().item()
    assert abs(zero_frac - p) < 0.02, f"the zeroed-out fraction should be ≈{p}, got {zero_frac:.3f}"
    # kept elements should be scaled up to 1/(1-p)
    kept = out[out != 0]
    assert torch.allclose(kept, torch.full_like(kept, 1 / (1 - p)), atol=1e-4)
    # the expectation should stay ≈ the original value 1
    assert abs(out.mean().item() - 1.0) < 0.02, "inverted dropout should keep the expectation unchanged"
