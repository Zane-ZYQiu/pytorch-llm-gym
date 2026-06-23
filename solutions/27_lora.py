HINTS = [
    "Base: base = x @ self.weight.t().",
    "Low-rank delta: lora = (x @ self.A.t()) @ self.B.t(). First x@A^T gives (...,r), then @B^T gives (...,out).",
    "return base + lora * self.scaling.",
    "Why x@A^T then @B^T instead of computing B@A first: it avoids explicitly building the big (out,in) matrix, saving memory.",
]

# ===== reference solution =====
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.t()
        lora = (x @ self.A.t()) @ self.B.t()
        return base + lora * self.scaling
