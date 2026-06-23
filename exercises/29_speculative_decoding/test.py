import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def test_acceptance_prob():
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.3, 0.7])
    assert torch.allclose(task.acceptance_prob(p, q, 0), torch.tensor(1.0))   # 0.6/0.3=2 -> 1
    assert torch.allclose(task.acceptance_prob(p, q, 1), torch.tensor(0.4 / 0.7), atol=1e-6)


def test_residual_dist():
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.3, 0.7])
    r = task.residual_dist(p, q)
    # max(p-q,0) = [0.3, 0] -> normalized [1, 0]
    assert torch.allclose(r, torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(r.sum(), torch.tensor(1.0), atol=1e-6)


def test_marginal_equals_target():
    """Core theorem: after accept/resample, the final token distribution should equal the target distribution p."""
    torch.manual_seed(0)
    V = 5
    p = torch.softmax(torch.randn(V), dim=-1)
    q = torch.softmax(torch.randn(V), dim=-1)
    r = task.residual_dist(p, q)

    N = 200_000
    drafts = torch.multinomial(q, N, replacement=True)
    us = torch.rand(N)
    residuals = torch.multinomial(r, N, replacement=True)

    counts = torch.zeros(V)
    for i in range(N):
        tok = task.speculative_sample(p, q, int(drafts[i]), float(us[i]), int(residuals[i]))
        counts[tok] += 1
    freq = counts / N
    assert torch.allclose(freq, p, atol=0.01), f"empirical distribution should ≈ target p\n{freq}\nvs\n{p}"
