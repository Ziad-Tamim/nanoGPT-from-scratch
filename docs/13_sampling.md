# 13 — Sampling / Text Generation

> **Code:** `src/nanogpt/generate.py` · **Diagram:** `assets/13_sampling.png`
> **Status:** stub — fill in during Step 12.

## Intuition

Generation is autoregressive: predict the next token, append it, feed the longer sequence
back in, repeat. The model writes one character at a time.

## Mechanics

```
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]          # crop to context window
    logits, _ = model(idx_cond)              # (B, T, vocab_size)
    logits = logits[:, -1, :] / temperature  # last step only
    # optional top-k: keep k largest logits, set rest to -inf
    probs = softmax(logits, dim=-1)
    next_id = multinomial(probs, 1)          # sample
    idx = cat([idx, next_id], dim=1)
```

## Knobs

- **temperature** > 1 = more random, < 1 = more confident/greedy. `→0` ≈ argmax.
- **top-k** restricts sampling to the k most likely tokens (cuts off the tail).
- Crop context to `block_size` — the model can't attend beyond what it was trained on.

## Mechanics note

Pure inference: `model.eval()` + `torch.no_grad()`. Decode ids back to text with the
tokenizer (doc 03).

## References

- Holtzman et al. (2020), top-k / nucleus sampling; Karpathy, nanoGPT `generate`.
