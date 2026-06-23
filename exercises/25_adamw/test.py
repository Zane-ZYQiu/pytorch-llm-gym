import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn
from practharness import load_task

task = load_task(__file__)


def test_matches_torch_adamw():
    torch.manual_seed(0)
    init = torch.randn(20)
    target = torch.randn(20)

    x_mine = nn.Parameter(init.clone())
    x_ref = nn.Parameter(init.clone())
    opt_mine = task.AdamW([x_mine], lr=1e-2, weight_decay=0.05)
    opt_ref = torch.optim.AdamW([x_ref], lr=1e-2, weight_decay=0.05)

    for _ in range(50):
        opt_mine.zero_grad()
        loss_mine = ((x_mine - target) ** 2).sum()
        loss_mine.backward()
        opt_mine.step()

        opt_ref.zero_grad()
        loss_ref = ((x_ref - target) ** 2).sum()
        loss_ref.backward()
        opt_ref.step()

    assert torch.allclose(x_mine.data, x_ref.data, atol=1e-5), \
        "your AdamW trajectory should match torch.optim.AdamW"


def test_minimizes_quadratic():
    torch.manual_seed(1)
    x = nn.Parameter(torch.randn(10))
    target = torch.zeros(10)
    opt = task.AdamW([x], lr=5e-2, weight_decay=0.0)
    first = ((x - target) ** 2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        loss = ((x - target) ** 2).sum()
        loss.backward()
        opt.step()
    assert loss.item() < first * 1e-2, "loss should drop significantly"
