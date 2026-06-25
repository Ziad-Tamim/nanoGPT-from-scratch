# 13 — Sampling / Text Generation

> **Code:** `src/nanogpt/generate.py` (+ `GPT.generate` in `model.py`) · **Diagram:** `assets/13_sampling.png`

## Intuition

The model only ever predicts **one** thing: a probability distribution over the next token
given the tokens so far. To produce text we use it autoregressively — sample a token from that
distribution, append it to the context, and ask again. The model literally writes one
character at a time, each new character feeding back in as input.

## Mechanics

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]      # crop to the context window
    logits, _ = model(idx_cond)          # (B, T, vocab)
    logits = logits[:, -1, :] / temperature   # keep only the last step -> (B, vocab)
    # optional top-k: keep the k largest logits, set the rest to -inf
    probs = softmax(logits, dim=-1)
    next_id = multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)
```

Two details that matter:

- **Crop to `block_size`.** The model has positions only for `block_size` tokens (its
  trained context). Past that we slide the window, keeping the most recent tokens.
- **Last step only.** A forward pass yields a prediction at *every* position, but when
  extending the sequence we only need the distribution at the final position.

## The sampling knobs

### Temperature

Divide logits by `τ` before softmax:

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.$$

- `τ → 0`: distribution collapses onto the argmax → greedy, repetitive, "safe".
- `τ = 1`: the model's true distribution.
- `τ > 1`: flatter distribution → more surprising, more mistakes.

A common sweet spot for this model is `τ ≈ 0.8`.

### Top-k

Before sampling, keep only the `k` highest-logit tokens and set the rest to `−∞` (zero
probability). This prunes the long tail of unlikely tokens that would otherwise occasionally
get sampled and derail the text. `k = 200` is a reasonable default for a 65-token vocab; small
`k` is more conservative.

## Greedy vs. sampling

Always taking the argmax (`τ→0`, or `k=1`) is deterministic but tends to loop and produce dull
text. Sampling with moderate temperature + top-k keeps output varied while staying coherent —
the standard choice for creative generation.

## Inference hygiene

Generation runs under `model.eval()` (dropout off) and `torch.no_grad()` (no autograd graph),
both handled for you: `load_checkpoint` sets eval mode and `GPT.generate` is decorated with
`@torch.no_grad()`. The checkpoint stores the tokenizer vocab, so decoding uses the exact same
char↔id mapping the model was trained with.

## Usage

```bash
uv run python -m nanogpt.generate --prompt "ROMEO:" --tokens 500 --temperature 0.8 --top-k 200
```

## References

- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration* (top-k / nucleus).
- Karpathy, *nanoGPT* (`generate`).
