# 07 — Multi-Head Attention

> **Code:** `src/nanogpt/multi_head_attention.py` · **Diagram:** `assets/07_multi_head_attention.png`

## Intuition

A single attention head produces one set of weights — it can only emphasize *one* kind of
relationship at a time (say, "attend to the previous word"). Language needs many at once:
agreement with the subject, matching brackets, topic, position. **Multi-head attention** runs
`n_head` independent heads in parallel, each in its own smaller subspace, so they can
specialize, then combines their results.

## Math / mechanics

Split the model width `C` into `h = n_head` heads of size `d = C / h`. Head `i` has its own
projections `W_Q^i, W_K^i, W_V^i ∈ ℝ^{C×d}` and runs scaled dot-product causal attention
(doc 06):

$$
\text{head}_i = \mathrm{Attention}_i(x) \in \mathbb{R}^{B\times T\times d}.
$$

Concatenate along the feature axis and apply a learned output projection `W_O ∈ ℝ^{C×C}`:

$$
\mathrm{MHA}(x) = \big[\text{head}_1 \,\Vert\, \cdots \,\Vert\, \text{head}_h\big]\,W_O
\in \mathbb{R}^{B\times T\times C}.
$$

The concatenation restores width `C`; `W_O` lets the model **mix** information across heads
before it re-enters the residual stream (doc 09). Dropout is applied after the projection.

### Why split instead of one big head

With `h` heads of size `C/h`, the total projection parameters are `3·C·(C/h)·h = 3C²` — the
**same** as one full-width head. The split costs nothing in parameters but gives the model `h`
separate attention patterns. Each head's `√d` scaling uses the *per-head* size `d = C/h`.

## Implementation note

We build `MultiHeadAttention` from a `ModuleList` of `Head` modules for readability. A fused
implementation does all heads in one matmul by projecting to `3C`, reshaping to
`(B, n_head, T, d)`, and batching the attention — identical math, faster on GPU. Worth doing
as a later optimization once the concept is clear.

## Shapes

| Tensor | Shape |
|---|---|
| `x` | `(B, T, C)` |
| each `head_i` | `(B, T, C/h)` |
| concat | `(B, T, C)` |
| after `W_O` | `(B, T, C)` |
| per-head weights (viz) | `(B, n_head, T, T)` |

## Backprop note

The heads are independent computational paths that **sum** their gradient contributions at
the shared input `x` (because each reads the same `x`). `W_O` receives gradient from the
concatenated head outputs and routes it back to the appropriate head slices. Causality is
preserved per head (tested in `test_each_head_is_causal` and end-to-end in
`test_causality_preserved_end_to_end`).

## References

- Vaswani et al. (2017), §3.2.2 (Multi-Head Attention).
