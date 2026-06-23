import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from practharness import load_task

task = load_task(__file__)


def _reference(image_emb, text_emb, temperature):
    i = F.normalize(image_emb, dim=-1)
    t = F.normalize(text_emb, dim=-1)
    logits = i @ t.t() / temperature
    labels = torch.arange(image_emb.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def test_matches_reference():
    img = torch.randn(8, 16)
    txt = torch.randn(8, 16)
    out = task.clip_loss(img, txt, temperature=0.07)
    assert out.shape == ()
    assert torch.allclose(out, _reference(img, txt, 0.07), atol=1e-5)


def test_perfect_alignment_low_loss():
    # image and text embeddings are identical and mutually distinguishable -> loss should be very small
    torch.manual_seed(0)
    emb = torch.randn(16, 32)
    aligned = task.clip_loss(emb, emb.clone(), temperature=0.07)
    random_txt = task.clip_loss(emb, torch.randn(16, 32), temperature=0.07)
    assert aligned < 0.1, f"loss for perfect alignment should be close to 0, got {aligned.item()}"
    assert aligned < random_txt


def test_symmetry():
    # swapping image/text should leave the symmetric loss unchanged
    img = torch.randn(6, 16)
    txt = torch.randn(6, 16)
    a = task.clip_loss(img, txt, 0.1)
    b = task.clip_loss(txt, img, 0.1)
    assert torch.allclose(a, b, atol=1e-5)
