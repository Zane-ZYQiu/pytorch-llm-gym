HINTS = [
    "pos = torch.arange(seq_len).unsqueeze(1).float()  # (L,1) column vector.",
    "Frequency denominator: div = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)  # (d/2,).",
    "pe = torch.zeros(seq_len, d_model); pe[:, 0::2] = torch.sin(pos / div); pe[:, 1::2] = torch.cos(pos / div).",
    "0::2 is the even columns, 1::2 is the odd columns. pos/div broadcasts to (L, d/2).",
]

# ===== reference solution =====
import torch


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len).unsqueeze(1).float()                 # (L, 1)
    div = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)  # (d/2,)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe
