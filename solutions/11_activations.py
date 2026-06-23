HINTS = [
    "silu: return x * torch.sigmoid(x).",
    "gelu: import math; return 0.5 * x * (1 + torch.erf(x / math.sqrt(2))).",
    "swiglu_ffn: gate = x @ W_gate; up = x @ W_up; h = silu(gate) * up; return h @ W_down.",
    "Intuition for the gate: silu(gate) acts as a soft switch deciding how much of each feature in up passes through.",
]

# ===== reference solution =====
import math
import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swiglu_ffn(x: torch.Tensor, W_gate: torch.Tensor,
               W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    gate = x @ W_gate
    up = x @ W_up
    h = silu(gate) * up
    return h @ W_down
