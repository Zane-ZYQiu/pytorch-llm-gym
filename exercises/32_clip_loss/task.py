"""32 · CLIP Contrastive Loss (symmetric InfoNCE)
Level: 6

CLIP aligns two modalities with image-text contrastive learning: within a batch, the i-th image and the i-th caption form a positive pair,
and all others are negatives. Maximize the similarity of positive pairs and suppress that of negative pairs.

  1. L2-normalize the image and text embeddings.
  2. Similarity matrix logits = (image @ text^T) / temperature   # (B, B)
  3. The correct pairs lie on the diagonal, so the labels are just 0..B-1.
  4. Symmetric loss = (CE(logits, labels) + CE(logits^T, labels)) / 2
     (both the image->text and text->image directions must classify correctly).

## Your task
Run: python practice.py check 32
"""
import torch
import torch.nn.functional as F


def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor,
              temperature: float = 0.07) -> torch.Tensor:
    """image_emb and text_emb are both (B, D). Returns a scalar symmetric contrastive loss."""
    raise NotImplementedError("TODO: normalize -> similarity/temperature -> average of the two-way cross-entropy")
