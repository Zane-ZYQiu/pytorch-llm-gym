import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def _block():
    torch.manual_seed(0)
    return task.TransformerBlock(d_model=16, num_heads=4, d_ff=32).eval()


def test_output_shape():
    blk = _block()
    x = torch.randn(2, 5, 16)
    out = blk(x)
    assert out.shape == (2, 5, 16)


def test_residual_identity_when_outputs_zeroed():
    """Zero out the final projection weights of both sublayers; the block should
    collapse to the identity (only the residual remains)."""
    blk = _block()
    with torch.no_grad():
        blk.Wo.weight.zero_()
        blk.W_down.weight.zero_()
    x = torch.randn(2, 5, 16)
    out = blk(x)
    assert torch.allclose(out, x, atol=1e-5), "the residual branch should collapse the block to the identity"


def test_causal_dependency():
    """Causality: changing token j should not affect outputs at positions < j."""
    blk = _block()
    x = torch.randn(1, 6, 16)
    out = blk(x)
    x2 = x.clone()
    x2[:, 3] += 10.0
    out2 = blk(x2)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5), "positions 0..2 should not be affected by position 3"
    assert not torch.allclose(out[:, 3], out2[:, 3], atol=1e-5), "position 3 should change"
