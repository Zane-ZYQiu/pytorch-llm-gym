import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def _ref_rmsnorm(x, gamma, eps=1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * gamma


def test_rms_norm_matches_reference():
    x = torch.randn(2, 4, 8)
    gamma = torch.randn(8)
    out = task.rms_norm(x, gamma, eps=1e-5)
    assert out.shape == x.shape
    assert torch.allclose(out, _ref_rmsnorm(x, gamma), atol=1e-5)


def test_rms_norm_no_mean_subtraction():
    # RMSNorm doesn't subtract the mean: given a same-sign vector, the RMS after scaling should be 1 (gamma=1)
    x = torch.full((1, 4), 3.0)
    gamma = torch.ones(4)
    out = task.rms_norm(x, gamma, eps=0.0)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(1), atol=1e-4)
    # and since there's no beta and no mean subtraction, the output stays a positive constant vector
    assert (out > 0).all()
