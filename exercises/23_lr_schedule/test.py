import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math
import torch
from practharness import load_task

task = load_task(__file__)


def test_warmup_linear():
    base = 1.0
    # warmup=10: step0->0, step5->0.5, step10->base
    assert abs(task.lr_at_step(0, base, 10, 100) - 0.0) < 1e-6
    assert abs(task.lr_at_step(5, base, 10, 100) - 0.5) < 1e-6
    assert abs(task.lr_at_step(10, base, 10, 100) - 1.0) < 1e-6


def test_cosine_decay():
    base, warm, total = 1.0, 10, 110
    # the cosine midpoint (progress=0.5) should be half of base
    mid = task.lr_at_step(10 + 50, base, warm, total, min_lr=0.0)
    assert abs(mid - 0.5) < 1e-6
    # at the end, progress=1, should reach min_lr
    end = task.lr_at_step(total, base, warm, total, min_lr=0.1)
    assert abs(end - 0.1) < 1e-6


def test_lr_monotonic_after_warmup():
    base, warm, total = 1.0, 10, 110
    vals = [task.lr_at_step(s, base, warm, total) for s in range(10, 111)]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)), "should be monotonically non-increasing after warmup"


def test_clip_grad_norm_scales():
    g1 = torch.tensor([3.0, 4.0])   # norm 5
    g2 = torch.tensor([0.0, 0.0])
    total = task.clip_grad_norm_([g1, g2], max_norm=1.0)
    assert torch.allclose(total, torch.tensor(5.0), atol=1e-5), "should return the norm before clipping"
    new_norm = torch.sqrt((g1**2).sum() + (g2**2).sum())
    assert torch.allclose(new_norm, torch.tensor(1.0), atol=1e-4), "norm after clipping should = max_norm"


def test_clip_grad_norm_noop_when_small():
    g = torch.tensor([0.3, 0.4])    # norm 0.5 < 1
    before = g.clone()
    total = task.clip_grad_norm_([g], max_norm=1.0)
    assert torch.allclose(total, torch.tensor(0.5), atol=1e-5)
    assert torch.allclose(g, before), "gradients should not change when the norm does not exceed the threshold"
