"""19 · Transformer Block (pre-norm + residual)
Level: 3

Assemble the components you've learned into a modern Transformer decoder block
(LLaMA style):

    x = x + Attention(RMSNorm(x))        # attention sublayer + residual
    x = x + SwiGLU_FFN(RMSNorm(x))       # feed-forward sublayer + residual

Note this is **pre-norm** (normalize first, then enter the sublayer), which is more
stable and easier to train than post-norm. The residual connection (x + ...) is a
gradient highway that makes deep networks trainable. Attention uses a causal mask.

The layers in __init__ are already provided (to make testing easy). You only need
to implement forward.

## Your task: implement TransformerBlock.forward
Run: python practice.py check 19
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        # attention projections (no bias)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        # SwiGLU FFN
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def _causal_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Optional helper: run causal multi-head attention on the already-normalized
        h (B,T,D) and return (B,T,D). You can implement it here, or write it directly
        into forward."""
        raise NotImplementedError("TODO (optional): multi-head causal attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, D) -> (B, T, D). pre-norm + residual, two sublayers."""
        raise NotImplementedError("TODO: x = x + Attn(norm1(x)); x = x + FFN(norm2(x))")
