# Roadmap

A progressive curriculum aligned with the most common "implement it from scratch" questions for
LLM / multimodal-LLM roles, organized in the spirit of *Alisa's Book of LLMs*.
`[x]` = built, `[ ]` = planned.

---

## Level 0 · Tensor & PyTorch basics
Rebuild your PyTorch muscle memory — all of these get reused later.
- [x] 01 Tensor basics: arange / reshape / view / transpose / **split_heads**
- [x] 02 Broadcasting & vectorization: kill the for-loops, pairwise distances, masked mean
- [x] 03 Indexing / gather / scatter / mask: **gather token log-probs**, one-hot, causal mask
- [x] 04 Reductions & statistics: mean/var/std (keepdim), standardize, **global grad norm**
- [x] 05 Autograd: requires_grad / backward / numerical gradient check

## Level 1 · Core NN building blocks
The classic "don't use nn.X, write it from scratch" questions.
- [x] 06 Linear layer: forward + **manual backward (grad_x / grad_W / grad_b)**
- [x] 07 Numerically stable softmax / log-softmax / logsumexp
- [x] 08 Cross-entropy loss + **gradient = softmax − onehot**
- [x] 09 LayerNorm from scratch
- [x] 10 RMSNorm from scratch
- [x] 11 Activations & **SwiGLU FFN** (GELU / SiLU)
- [x] 12 Dropout (inverted dropout, train/eval)

## Level 2 · Attention ⭐ interview core
- [x] 13 Scaled dot-product attention + causal mask (single head)
- [x] 14 Multi-head attention (full reshape/transpose flow)
- [x] 15 Grouped-query attention GQA / MQA
- [x] 16 KV cache (incremental decoding equals one-shot causal)

## Level 3 · Positional encoding & Transformer
- [x] 17 Sinusoidal positional encoding
- [x] 18 RoPE (verifies relative-position invariance)
- [x] 19 Transformer block (pre-norm + residual + causality)
- [x] 20 Full GPT decoder LM (weight tying + greedy generate)

## Level 4 · Training & generation
- [x] 21 Training loop + teacher forcing (actually overfits toy data)
- [x] 22 Sampling: greedy / temperature / top-k / top-p
- [x] 23 LR schedule: warmup + cosine; gradient clipping
- [x] 24 Beam search

## Level 5 · Advanced LLM
- [x] 25 AdamW from scratch (matches torch.optim.AdamW)
- [x] 26 Mixture of Experts (top-k routing)
- [x] 27 LoRA linear (frozen base + zero-init B)
- [x] 28 Online softmax / Flash-Attention tiling (exact, not approximate)
- [x] 29 Speculative decoding accept/resample (Monte-Carlo verified theorem)
- [x] 30 RL alignment losses: DPO / PPO clip / GRPO advantage

## Level 6 · Multimodal ⭐⭐
- [x] 31 ViT patch embedding: patchify + linear projection + CLS + position
- [x] 32 CLIP contrastive loss (symmetric InfoNCE)
- [x] 33 LLaVA-style projector + image-text token fusion
- [x] 34 Cross-modal cross-attention fusion (image features as K/V)
- [x] 35 2D-RoPE: variable-resolution visual positional encoding (verifies 2D relative invariance)

---

### Status
**All 35 problems (Level 0–6) are built and every reference solution passes its own tests.**
This already covers the large majority of the "implement it from scratch" questions for
multimodal-LLM roles.

### Ideas for later
- Harder variants: sliding-window attention, MLA, ALiBi, Gumbel-Softmax, straight-through estimator
- Per-problem deep-dive write-ups and more interactive explainers
- A hosted online version (needs a sandboxed code-execution backend)
