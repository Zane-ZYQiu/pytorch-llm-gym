"""27 · LoRA Linear Layer (Low-Rank Adaptation)
Level: 5

Fine-tuning large models with full parameter training is too expensive. LoRA freezes the original weight W and trains only a low-rank delta ΔW = B·A:

    y = x W^T + (x A^T) B^T · (alpha / r)

  • W: (out, in) frozen pretrained weight
  • A: (r, in), B: (out, r), rank r << min(in,out); only A and B are trainable
  • scaling = alpha / r
  • **B is initialized to 0** -> at the start of training ΔW=0, so the output == the original model (doesn't break existing capabilities)

Interview point: trainable parameters drop from in×out to r×(in+out), saving several orders of magnitude.

__init__ is given (with B zero-init and W frozen). You implement forward.

## Your task
Run: python practice.py check 27
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad_(False)               # freeze pretrained weight
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))   # zero init
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., in) -> (..., out). = frozen base output + low-rank delta * scaling."""
        raise NotImplementedError("TODO: x@W^T + (x@A^T)@B^T * scaling")
