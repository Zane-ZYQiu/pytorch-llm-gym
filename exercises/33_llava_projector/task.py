"""33 · LLaVA-style Projector + Image-Text Fusion
Level: 6

How does LLaVA connect visual features into an LLM? Crude but effective (early fusion / concatenation):
  1. The vision encoder (CLIP) outputs image features (N_img, D_vision).
  2. An **MLP projector** maps them to the LLM's word-embedding dimension D_text (independently per token):
        h = GELU(image_feat @ W1 + b1) @ W2 + b2     # (N_img, D_text)
  3. The projected visual tokens are simply [prepended to the text word embeddings] and fed to the LLM as an ordinary token sequence:
        sequence = concat([visual_tokens, text_embeds], dim=sequence_dim)

Interview note: compared with Q-Former/cross-attention, LLaVA uses the simplest "linear projection + concatenation",
which is training-efficient, works well, and is currently mainstream.

## Your task
Run: python practice.py check 33
"""
import torch
import torch.nn.functional as F


def build_multimodal_input(image_features: torch.Tensor, text_embeds: torch.Tensor,
                           W1: torch.Tensor, b1: torch.Tensor,
                           W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """image_features (B, N_img, D_vision); text_embeds (B, N_txt, D_text).
    W1 (D_vision, D_hidden), b1 (D_hidden,), W2 (D_hidden, D_text), b2 (D_text,).
    Returns the concatenated sequence (B, N_img + N_txt, D_text), with visual tokens first."""
    raise NotImplementedError("TODO: two-layer GELU-MLP projection -> concatenate with text embeddings")
