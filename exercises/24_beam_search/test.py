import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools
import torch
from practharness import load_task

task = load_task(__file__)

V = 3
torch.manual_seed(1)
M = torch.log_softmax(torch.randn(V, V), dim=-1)  # each row is a logprob distribution (Markov transition)
START = 0


def next_logprobs(seq):
    return M[seq[-1].item()]


def _brute_force(max_len):
    best_seq, best = None, -1e9
    for combo in itertools.product(range(V), repeat=max_len):
        seq, score = [START], 0.0
        for tok in combo:
            score += M[seq[-1], tok].item()
            seq.append(tok)
        if score > best:
            best, best_seq = score, seq
    return best_seq


def test_returns_prefix_and_length():
    out = task.beam_search(next_logprobs, START, num_beams=2, max_len=3)
    assert out[0] == START
    assert len(out) == 4  # start + 3 new tokens


def test_greedy_when_one_beam():
    out = task.beam_search(next_logprobs, START, num_beams=1, max_len=3)
    # num_beams=1 is greedy
    cur = [START]
    for _ in range(3):
        cur.append(int(next_logprobs(torch.tensor(cur)).argmax()))
    assert out == cur


def test_exhaustive_equals_brute_force():
    # when num_beams is large enough it is exhaustive, should give the global optimum
    out = task.beam_search(next_logprobs, START, num_beams=V**3, max_len=3)
    assert out == _brute_force(3)


def test_eos_stops_expansion():
    # setup: token 1 is eos, and from 0 the most likely next is 1
    Mloc = torch.log(torch.tensor([[0.1, 0.8, 0.1],
                                   [0.34, 0.33, 0.33],
                                   [0.34, 0.33, 0.33]]))

    def lp(seq):
        return Mloc[seq[-1].item()]

    out = task.beam_search(lp, 0, num_beams=2, max_len=5, eos_token=1)
    assert 1 in out, "should generate up to eos"
