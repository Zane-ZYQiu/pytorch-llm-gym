import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_greedy():
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 1.0, 2.0]])
    out = task.greedy(logits)
    assert torch.equal(out, torch.tensor([1, 0]))


def test_top_k_filter():
    torch.manual_seed(0)
    logits = torch.randn(3, 10)
    out = task.top_k_filter(logits, k=3)
    assert out.shape == logits.shape
    finite = torch.isfinite(out)
    assert torch.equal(finite.sum(-1), torch.full((3,), 3)), "each row should keep exactly 3"
    # the kept ones should be the 3 largest
    topk_idx = torch.topk(logits, 3, dim=-1).indices
    for r in range(3):
        kept = finite[r].nonzero().flatten()
        assert set(kept.tolist()) == set(topk_idx[r].tolist())


def test_top_p_filter():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = torch.log(probs)
    out = task.top_p_filter(logits, p=0.79)
    finite = torch.isfinite(out)[0]
    # 0.5 + 0.3 = 0.8 >= 0.79, should keep the first two
    assert finite.tolist() == [True, True, False, False]


def test_top_p_keeps_at_least_one():
    probs = torch.tensor([[0.99, 0.005, 0.005]])
    logits = torch.log(probs)
    out = task.top_p_filter(logits, p=0.5)
    finite = torch.isfinite(out)[0]
    assert finite[0].item() is True and finite.sum().item() == 1, "even with a tiny p, keep at least the largest one"
