HINTS = [
    "projection layer 1: h = F.gelu(image_features @ W1 + b1)  # (B, N_img, D_hidden).",
    "projection layer 2: visual = h @ W2 + b2  # (B, N_img, D_text).",
    "concatenate: return torch.cat([visual, text_embeds], dim=1)  # visual first, along the sequence dim (dim=1).",
    "note broadcasting: b1 of shape (D_hidden,) is automatically added to the last dim of (B,N_img,D_hidden).",
]

# ===== reference solution =====
import torch
import torch.nn.functional as F


def build_multimodal_input(image_features: torch.Tensor, text_embeds: torch.Tensor,
                           W1: torch.Tensor, b1: torch.Tensor,
                           W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    visual = F.gelu(image_features @ W1 + b1) @ W2 + b2     # (B, N_img, D_text)
    return torch.cat([visual, text_embeds], dim=1)
