import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch
from practharness import load_task

task = load_task(__file__)


def _gpt():
    torch.manual_seed(0)
    return task.GPT(vocab_size=20, d_model=16, num_heads=4,
                    d_ff=32, num_layers=2, max_len=32).eval()


def test_forward_shape():
    gpt = _gpt()
    idx = torch.randint(0, 20, (2, 7))
    logits = gpt(idx)
    assert logits.shape == (2, 7, 20)


def test_weight_tying():
    gpt = _gpt()
    assert gpt.lm_head.weight.data_ptr() == gpt.tok_emb.weight.data_ptr(), \
        "lm_head should share the same weight table as the token embedding"


def test_causal_logits():
    gpt = _gpt()
    idx = torch.randint(0, 20, (1, 8))
    logits = gpt(idx)
    idx2 = idx.clone()
    idx2[0, 5] = (idx2[0, 5] + 1) % 20  # change token 5
    logits2 = gpt(idx2)
    assert torch.allclose(logits[:, :5], logits2[:, :5], atol=1e-5), \
        "the logits at positions 0..4 should not be affected by position 5 (causal)"


def test_generate_greedy():
    gpt = _gpt()
    idx = torch.randint(0, 20, (1, 4))
    out = gpt.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], idx), "the original prefix should be preserved"
    # manually reproduce greedy decoding; the result should match
    cur = idx.clone()
    for _ in range(5):
        nxt = gpt(cur)[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    assert torch.equal(out, cur), "generate should be equivalent to taking argmax step by step"
