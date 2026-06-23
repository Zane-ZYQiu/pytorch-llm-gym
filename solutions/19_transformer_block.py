HINTS = [
    "forward skeleton: h = self.norm1(x); a = causal multi-head attention(h); x = x + self.Wo(the merged-head result of a) (in practice put Wo inside the attention); then h2 = self.norm2(x); x = x + FFN(h2).",
    "Attention: B,T,D=h.shape; H=self.num_heads; hd=D//H. q=self.Wq(h).reshape(B,T,H,hd).transpose(1,2), and k, v likewise.",
    "scores=q@k.transpose(-2,-1)/sqrt(hd); mask=triu(ones(T,T,bool),1); scores.masked_fill(mask,-inf); attn=softmax; o=(attn@v).transpose(1,2).reshape(B,T,D); attn_out=self.Wo(o).",
    "FFN: ff = self.W_down(F.silu(self.W_gate(h2)) * self.W_up(h2)). Don't forget the x = x + ... residual in both places.",
]

# ===== reference solution =====
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
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def _causal_attention(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        H = self.num_heads
        hd = D // H
        q = self.Wq(h).reshape(B, T, H, hd).transpose(1, 2)
        k = self.Wk(h).reshape(B, T, H, hd).transpose(1, 2)
        v = self.Wv(h).reshape(B, T, H, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.Wo(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._causal_attention(self.norm1(x))
        h2 = self.norm2(x)
        ff = self.W_down(F.silu(self.W_gate(h2)) * self.W_up(h2))
        x = x + ff
        return x
