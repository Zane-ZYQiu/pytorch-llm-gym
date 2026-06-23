import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn as nn
from practharness import load_task

task = load_task(__file__)


def test_shift_for_lm():
    tokens = torch.arange(2 * 6).reshape(2, 6)
    inp, tgt = task.shift_for_lm(tokens)
    assert inp.shape == (2, 5) and tgt.shape == (2, 5)
    assert torch.equal(inp, tokens[:, :-1])
    assert torch.equal(tgt, tokens[:, 1:])


def test_lm_loss_matches_manual():
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    out = task.lm_loss(logits, targets)
    ref = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    assert torch.allclose(out, ref, atol=1e-5)


class _TinyLM(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        return self.head(self.emb(x))


def test_train_step_overfits():
    torch.manual_seed(0)
    V, D = 12, 16
    model = _TinyLM(V, D)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tokens = torch.randint(0, V, (4, 9))
    inp, tgt = task.shift_for_lm(tokens)

    first = task.train_step(model, inp, tgt, opt)
    for _ in range(60):
        last = task.train_step(model, inp, tgt, opt)
    assert isinstance(first, float)
    assert last < first * 0.5, f"repeatedly training the same batch should clearly overfit: {first:.3f} -> {last:.3f}"
