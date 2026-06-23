import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_gather_token_logprobs():
    B, T, V = 2, 4, 7
    logprobs = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    out = task.gather_token_logprobs(logprobs, targets)
    assert out.shape == (B, T)
    for b in range(B):
        for t in range(T):
            assert torch.allclose(out[b, t], logprobs[b, t, targets[b, t]])


def test_one_hot_scatter():
    idx = torch.tensor([0, 2, 1, 2])
    out = task.one_hot_scatter(idx, num_classes=3)
    assert out.shape == (4, 3)
    assert out.dtype == torch.float32
    assert torch.equal(out, F.one_hot(idx, 3).float())


def test_apply_causal_mask_2d():
    scores = torch.zeros(3, 3)
    out = task.apply_causal_mask(scores)
    neg_inf = float("-inf")
    # lower triangle (including diagonal) stays 0, upper triangle is -inf
    for i in range(3):
        for j in range(3):
            if j > i:
                assert out[i, j] == neg_inf, f"position ({i},{j}) should be -inf"
            else:
                assert out[i, j] == 0.0, f"position ({i},{j}) should stay 0"


def test_apply_causal_mask_batched():
    scores = torch.randn(2, 4, 5, 5)  # (B, H, T, T)
    out = task.apply_causal_mask(scores)
    assert out.shape == scores.shape
    # after softmax, future positions should have probability 0
    probs = torch.softmax(out, dim=-1)
    assert torch.allclose(probs[..., 0, 1:], torch.zeros_like(probs[..., 0, 1:]))
