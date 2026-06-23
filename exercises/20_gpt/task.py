"""20 · A complete GPT decoder language model
Level: 3

Wire everything together into a GPT that produces logits and can generate:

    token id --(embedding)--> + position embedding --> [N TransformerBlocks] --> final RMSNorm
             --> lm_head --> logits (B, T, vocab)

Two interview points:
  1. **weight tying**: the lm_head weights share the same table as the token embedding
     (self.lm_head.weight = self.tok_emb.weight), saving parameters and often improving quality.
  2. **greedy generation (generate)**: each step takes argmax over the last position of
     logits, appends it to the end of the sequence, and loops.

The _Block below is already provided (the same one as the previous exercise). You implement:
  • the weight tying in __init__ (marked TODO)
  • forward
  • generate (greedy)

## Your task
Run: python practice.py check 20
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    """A pre-norm causal Transformer block (already implemented, use as-is)."""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.num_heads, D // self.num_heads
        h = self.norm1(x)
        q = self.Wq(h).reshape(B, T, H, hd).transpose(1, 2)
        k = self.Wk(h).reshape(B, T, H, hd).transpose(1, 2)
        v = self.Wv(h).reshape(B, T, H, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
        o = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.Wo(o)
        h2 = self.norm2(x)
        x = x + self.W_down(F.silu(self.W_gate(h2)) * self.W_up(h2))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, num_heads, d_ff)
                                     for _ in range(num_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # TODO: weight tying -- make self.lm_head.weight and self.tok_emb.weight the same table

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx (B, T) are long token ids. Return logits (B, T, vocab_size)."""
        raise NotImplementedError("TODO: embedding+position -> through blocks -> norm_f -> lm_head")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation. idx (B, T0), returns (B, T0 + max_new_tokens)."""
        raise NotImplementedError("TODO: loop max_new_tokens times, each step take argmax of the last position and concatenate")
