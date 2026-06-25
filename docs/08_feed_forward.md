# 08 — Feed-Forward Network (MLP)

> **Code:** `src/nanogpt/feed_forward.py` · **Diagram:** `assets/08_feed_forward.png`
> **Status:** stub — fill in during Step 7.

## Intuition

Attention mixes information *across* positions; the feed-forward network processes each
position *independently*, giving the model room to "think" non-linearly about what attention
gathered. Applied identically to every position (position-wise).

## Math / mechanics

```
FFN(x) = Dropout( W2 · GELU(W1 x + b1) + b2 )
```

- `W1 ∈ ℝ^{C × 4C}` expands, `W2 ∈ ℝ^{4C × C}` projects back. The 4× hidden width is the
  standard expansion ratio.
- Non-linearity: **GELU** (GPT-2 style). Note original Transformer used ReLU; discuss the
  difference.

## Shapes

`(B, T, C)` → `(B, T, 4C)` → `(B, T, C)`. No mixing across `T`.

## Backprop note

Pure MLP — gradients are standard linear + GELU derivatives. This block holds a large share
of the model's parameters (≈ 2 · 4C² per layer).

## References

- Vaswani et al. (2017), §3.3; Hendrycks & Gimpel (2016), GELU.
