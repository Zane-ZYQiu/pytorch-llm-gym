import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def _reference(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


def test_shape():
    pe = task.sinusoidal_pe(10, 16)
    assert pe.shape == (10, 16)


def test_values():
    pe = task.sinusoidal_pe(12, 8)
    assert torch.allclose(pe, _reference(12, 8), atol=1e-5)


def test_first_position():
    # pos=0: sin(0)=0, cos(0)=1 -> even columns all 0, odd columns all 1
    pe = task.sinusoidal_pe(3, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)
