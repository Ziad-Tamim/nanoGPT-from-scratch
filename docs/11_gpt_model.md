# 11 — The GPT Model (assembly)

> **Code:** `src/nanogpt/model.py` · **Diagram:** `assets/11_gpt_model.png`
> **Status:** stub — fill in during Step 10.

## Intuition

Wire everything together: embed tokens, add positions, run `n_layer` blocks, normalize, and
project to vocabulary logits. The forward pass also computes the training loss.

## Math / mechanics

```
tok = TokenEmbedding(idx)          # (B, T, C)
x   = tok + PositionalEncoding     # (B, T, C)
x   = Blocks(x)                    # (B, T, C), ×n_layer
x   = LayerNorm(x)                 # final norm
logits = x · Wₕₑₐ𝒹                 # (B, T, vocab_size)
loss   = cross_entropy(logits, targets)
```

- **LM head** `W_head ∈ ℝ^{C × vocab_size}`. Optionally **tie weights** with the token
  embedding (share the matrix) — common in GPT; note pros/cons.
- **Cross-entropy** over the vocab at every position: `loss = −(1/BT) Σ log p(target)`.
  Reshape logits to `(B·T, vocab_size)` and targets to `(B·T,)`.

## Shapes

| Tensor | Shape |
|---|---|
| logits | `(B, T, vocab_size)` |
| targets | `(B, T)` |
| loss | scalar |

## Backprop note

Cross-entropy + softmax has the clean gradient `∂L/∂logits = (softmax(logits) − onehot)/N`.
Trace how this flows back through the LM head, blocks, and embeddings.

## References

- Radford et al. (2018/2019), GPT / GPT-2.
