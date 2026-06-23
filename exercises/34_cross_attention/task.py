"""34 · Cross-Modal Cross-Attention Fusion
Level: 6

Another kind of multimodal fusion (Flamingo/BLIP-2 style): let text queries "look at" image features.
The only difference from self-attention: **Q comes from one sequence, K/V come from another sequence (context)**.

  Q = x @ Wq           # query from the text/decode sequence (B, Tx, D)
  K = context @ Wk     # key/value from the image features (B, Tc, D)
  V = context @ Wv
  Then it's ordinary multi-head attention (no causal mask, since we're "looking at" the whole image).

This way each text position can retrieve relevant information from the image.

Convention: Wq/Wk/Wv/Wo are all (D, D).

## Your task
Run: python practice.py check 34
"""
import math
import torch


def cross_attention(x: torch.Tensor, context: torch.Tensor,
                    Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                    Wo: torch.Tensor, num_heads: int) -> torch.Tensor:
    """x (B, Tx, D) provides the query; context (B, Tc, D) provides key/value.
    Returns (B, Tx, D). Multi-head, no causal mask."""
    raise NotImplementedError("TODO: q from x, k/v from context, the rest is the same as multi-head attention")
