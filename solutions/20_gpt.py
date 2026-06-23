HINTS = [
    "Weight tying: at the end of __init__ write self.lm_head.weight = self.tok_emb.weight. This makes the two the same Parameter.",
    "forward: B,T = idx.shape; pos = torch.arange(T, device=idx.device); x = self.tok_emb(idx) + self.pos_emb(pos).",
    "for blk in self.blocks: x = blk(x); x = self.norm_f(x); return self.lm_head(x).",
    "generate: loop max_new_tokens times: logits = self(idx); nxt = logits[:, -1].argmax(-1, keepdim=True); idx = torch.cat([idx, nxt], dim=1). Finally return idx.",
]

# ===== reference solution =====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
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
        self.lm_head.weight = self.tok_emb.weight   # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
