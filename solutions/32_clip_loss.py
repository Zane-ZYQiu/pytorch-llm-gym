HINTS = [
    "normalize: i = F.normalize(image_emb, dim=-1); t = F.normalize(text_emb, dim=-1).",
    "similarity: logits = i @ t.t() / temperature  # (B,B), the diagonal holds the positive pairs.",
    "labels = torch.arange(B, device=...).",
    "symmetric: loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2. logits.t() corresponds to the text->image direction.",
]

# ===== reference solution =====
import torch
import torch.nn.functional as F


def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor,
              temperature: float = 0.07) -> torch.Tensor:
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.t() / temperature      # (B, B)
    labels = torch.arange(image_emb.shape[0], device=image_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
