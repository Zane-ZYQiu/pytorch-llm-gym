import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_layer_norm_matches_torch():
    x = torch.randn(2, 4, 8)
    gamma = torch.randn(8)
    beta = torch.randn(8)
    out = task.layer_norm(x, gamma, beta, eps=1e-5)
    ref = F.layer_norm(x, (8,), weight=gamma, bias=beta, eps=1e-5)
    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_layer_norm_identity_affine():
    # with gamma=1, beta=0, the output along the last dim should have mean≈0, var≈1
    x = torch.randn(3, 16) * 5 + 2
    gamma = torch.ones(16)
    beta = torch.zeros(16)
    out = task.layer_norm(x, gamma, beta)
    assert torch.allclose(out.mean(-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(3), atol=1e-2)
