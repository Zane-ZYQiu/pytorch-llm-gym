"""11 · Activation Functions and SwiGLU FFN
Level: 1

Modern LLM feed-forward layers (FFN) use SwiGLU, which is stronger than a plain ReLU-MLP.
First, prepare two activations:
  • SiLU / Swish:  silu(x) = x * sigmoid(x)
  • GELU (exact):  gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

SwiGLU FFN (a gated FFN):
  • gate = x @ W_gate           # (..., F)
  • up   = x @ W_up             # (..., F)
  • h    = silu(gate) * up      # element-wise product, the "gate"
  • out  = h @ W_down           # (..., D)
Note: here W_gate, W_up have shape (D, F) and W_down has shape (F, D), used as x @ W
(the plain matrix convention, not nn.Linear's transposed convention).

## Your task
Run: python practice.py check 11
"""
import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish: x * sigmoid(x)."""
    raise NotImplementedError("TODO: x * torch.sigmoid(x)")


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    raise NotImplementedError("TODO: use torch.erf")


def swiglu_ffn(x: torch.Tensor, W_gate: torch.Tensor,
               W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """x shape (..., D). W_gate/W_up shape (D, F), W_down shape (F, D).
    Return (..., D)."""
    raise NotImplementedError("TODO: silu(x@W_gate) * (x@W_up) then @ W_down")
