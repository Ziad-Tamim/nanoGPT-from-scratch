# 06 — Self-Attention (single head)

> **Code:** `src/nanogpt/attention.py` · **Diagram:** `assets/06_self_attention.png`
> **Status:** stub — fill in during Step 5. **This is the core of the Transformer.**

## Intuition

Each token builds a **query** ("what am I looking for"), and every token offers a **key**
("what I am") and a **value** ("what I'll pass on if attended to"). A token's output is a
weighted average of all values, weighted by query·key similarity. This is how tokens
communicate.

## Math / mechanics

From input `x ∈ (B, T, C)`, project with learnable weights:

```
Q = x Wq   K = x Wk   V = x Wv      # each (B, T, head_size)
scores      = Q Kᵀ / √(head_size)   # (B, T, T) — similarity of every pair
masked      = scores.masked_fill(causal_mask == 0, -inf)
weights     = softmax(masked, dim=-1)   # rows sum to 1
out         = weights V              # (B, T, head_size)
```

- **√(head_size) scaling** keeps the dot products from growing with dimension, which would
  otherwise push softmax into tiny-gradient saturation. (Derive the variance argument here.)
- **Causal mask** is lower-triangular: position `t` may attend only to `≤ t`. Setting future
  scores to −∞ makes their softmax weight 0. This is the "Masked" in *Masked* Multi-Head
  Attention and is what makes it a language model.

## Shapes

| Tensor | Shape |
|---|---|
| Q, K, V | `(B, T, head_size)` |
| scores / weights | `(B, T, T)` |
| out | `(B, T, head_size)` |

## Backprop note

Softmax Jacobian: `∂softmax_i/∂z_j = softmax_i(δ_ij − softmax_j)`. Discuss why the scaling
keeps these gradients healthy, and that the masked (−∞) entries get exactly zero gradient.

## References

- Vaswani et al. (2017), §3.2.1 (Scaled Dot-Product Attention).
