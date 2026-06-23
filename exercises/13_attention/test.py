import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def test_attention_shape_and_value():
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 16)
    out = task.scaled_dot_product_attention(q, k, v, causal=False)
    assert out.shape == (2, 4, 16)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_weights_sum_to_one():
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 5, 8)
    v = torch.eye(5).unsqueeze(0)  # value = identity matrix, so output equals the attention weights
    out = task.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out.sum(-1), torch.ones(1, 3), atol=1e-5), "each query's weights should sum to 1"


def test_attention_causal():
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    out = task.scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5), "causal attention result should match torch"
