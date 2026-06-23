"""17 · Sinusoidal Positional Encoding
Level: 3

Attention is itself position-blind (shuffling the token order gives the same result),
so we have to inject position information. The original Transformer uses fixed
sine/cosine encodings:

    PE[pos, 2i]   = sin( pos / 10000^(2i/d) )
    PE[pos, 2i+1] = cos( pos / 10000^(2i/d) )

Each dimension i corresponds to a different frequency: low dims are high-frequency,
high dims are low-frequency, and together they uniquely encode every position.
The benefits: nothing to learn, and it extrapolates to longer sequences.

## Your task
Run: python practice.py check 17
"""
import torch


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Return the (seq_len, d_model) positional encoding. d_model is even.
    Even columns hold sin, odd columns hold cos."""
    raise NotImplementedError("TODO: build pos and the frequencies, fill even/odd columns with sin/cos")
