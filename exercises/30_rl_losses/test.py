import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math
import torch
from practharness import load_task

task = load_task(__file__)


def test_dpo_equal_margins():
    # chosen and rejected have equal "relative-to-reference" margins -> logits=0 -> loss=-log(0.5)
    z = torch.zeros(4)
    loss = task.dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-5)


def test_dpo_prefers_chosen():
    # when chosen is clearly better than rejected the loss should be smaller
    pi_c = torch.tensor([2.0]); pi_r = torch.tensor([-2.0])
    ref = torch.tensor([0.0])
    good = task.dpo_loss(pi_c, pi_r, ref, ref, beta=1.0)
    bad = task.dpo_loss(pi_r, pi_c, ref, ref, beta=1.0)
    assert good < bad


def test_ppo_ratio_one():
    logp = torch.zeros(3); old = torch.zeros(3)
    adv = torch.tensor([1.0, -2.0, 3.0])
    loss = task.ppo_clip_loss(logp, old, adv, eps=0.2)
    assert torch.allclose(loss, -adv.mean(), atol=1e-6)


def test_ppo_clipping_positive_adv():
    # ratio=2, adv=+1, eps=0.2 -> objective=min(2, 1.2)=1.2 -> loss=-objective=-1.2
    logp = torch.tensor([math.log(2.0)]); old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = task.ppo_clip_loss(logp, old, adv, eps=0.2)
    assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-5)


def test_grpo_advantages():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    adv = task.grpo_advantages(rewards)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    expected = (rewards - mean) / (std + 1e-4)
    assert torch.allclose(adv, expected, atol=1e-5)
    assert torch.allclose(adv.mean(dim=-1), torch.zeros(1), atol=1e-5), "within-group mean should ≈ 0"
