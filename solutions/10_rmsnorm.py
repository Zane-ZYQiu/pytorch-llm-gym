HINTS = [
    "ms = x.pow(2).mean(dim=-1, keepdim=True)  # mean square, along the last dim.",
    "rms = torch.sqrt(ms + eps).",
    "return x / rms * gamma. Note there's no mean subtraction and no beta——that's the difference from LayerNorm.",
    "Advanced: you can also use x * torch.rsqrt(ms + eps) * gamma; rsqrt is 1/sqrt, which is faster.",
]

# ===== reference solution =====
import torch


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * gamma
