import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def _reference(x, Wq, Wk, Wv, Wo, N, K, causal):
    B, T, D = x.shape
    hd = D // N
    q = (x @ Wq).reshape(B, T, N, hd).transpose(1, 2)            # (B,N,T,hd)
    k = (x @ Wk).reshape(B, T, K, hd).transpose(1, 2)            # (B,K,T,hd)
    v = (x @ Wv).reshape(B, T, K, hd).transpose(1, 2)
    rep = N // K
    # explicitly pick, for each query head, the KV head of the group it belongs to
    head_to_kv = torch.arange(N) // rep                          # (N,)
    k = k[:, head_to_kv]                                         # (B,N,T,hd)
    v = v[:, head_to_kv]
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    o = o.transpose(1, 2).reshape(B, T, D)
    return o @ Wo


def _make(D, N, K):
    Wq = torch.randn(D, D)
    Wk = torch.randn(D, (D // N) * K)
    Wv = torch.randn(D, (D // N) * K)
    Wo = torch.randn(D, D)
    return Wq, Wk, Wv, Wo


def test_gqa_shape():
    x = torch.randn(2, 5, 16)
    Wq, Wk, Wv, Wo = _make(16, 4, 2)
    out = task.grouped_query_attention(x, Wq, Wk, Wv, Wo, num_heads=4, num_kv_heads=2)
    assert out.shape == (2, 5, 16)


def test_gqa_matches_reference():
    x = torch.randn(2, 6, 16)
    Wq, Wk, Wv, Wo = _make(16, 4, 2)
    out = task.grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 2, causal=True)
    ref = _reference(x, Wq, Wk, Wv, Wo, 4, 2, True)
    assert torch.allclose(out, ref, atol=1e-4)


def test_mqa_special_case():
    # K=1 is MQA: all query heads share a single KV head
    x = torch.randn(2, 6, 16)
    Wq, Wk, Wv, Wo = _make(16, 4, 1)
    out = task.grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 1, causal=True)
    ref = _reference(x, Wq, Wk, Wv, Wo, 4, 1, True)
    assert torch.allclose(out, ref, atol=1e-4)
