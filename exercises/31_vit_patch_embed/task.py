"""31 · ViT Patch Embedding
Level: 6

Vision Transformer feeds an image to the Transformer as a "sequence of tokens". The first step is patching:
  • Split a (C,H,W) image into non-overlapping patches (each p×p), giving N=(H/p)·(W/p) patches.
  • Flatten each patch into a vector of length C·p·p.
  • Linearly project to D dimensions -> N patch embeddings.
  • Prepend a learnable [CLS] token, then add positional embeddings -> (N+1, D) sequence.

This is exactly the entry point of the vision encoder in CLIP/LLaVA.

Convention: patches are in row-major order (row before column); within each patch, flatten in (C, p, p) order.

## Your task
Run: python practice.py check 31
"""
import torch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """images (B, C, H, W), with H and W divisible by patch.
    Returns (B, N, C*patch*patch), N=(H/patch)*(W/patch), patches in row-major order."""
    raise NotImplementedError("TODO: reshape to (B,C,nh,p,nw,p) -> permute -> reshape")


def vit_embed(images: torch.Tensor, patch: int, proj: torch.Tensor,
              cls: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """patchify -> linear projection -> prepend CLS -> add positional embeddings.
    proj (C*patch*patch, D); cls (1,1,D); pos (1, N+1, D). Returns (B, N+1, D)."""
    raise NotImplementedError("TODO: patchify -> @proj -> cat(cls) -> +pos")
