# 07 — Multi-Head Attention

> **Code:** `src/nanogpt/multi_head_attention.py` · **Diagram:** `assets/07_multi_head_attention.png`
> **Status:** stub — fill in during Step 6.

## Intuition

One attention head can only learn one "relationship pattern" at a time. Run `n_head` heads in
parallel, each in a smaller subspace, so the model attends to different things at once (e.g.
syntax vs. long-range reference), then combine them.

## Math / mechanics

- Split `C` into `n_head` heads of size `head_size = C / n_head`.
- Each head runs scaled dot-product attention (doc 06) independently.
- Concatenate the heads back to `(B, T, C)`, then a learnable output projection `Wo`.

```
head_i = Attention(x Wq_i, x Wk_i, x Wv_i)
MHA(x) = Concat(head_1, …, head_h) Wo
```

Efficient implementation: one big projection to `3·C`, reshape to
`(B, n_head, T, head_size)`, do batched attention, reshape back.

## Shapes

| Tensor | Shape |
|---|---|
| per-head out | `(B, T, head_size)` |
| concatenated | `(B, T, C)` |
| after `Wo` | `(B, T, C)` |

## Backprop note

Heads are independent paths that sum their gradient contributions at the shared input `x`;
`Wo` mixes them. Total params are similar to a single full-width head, but the split lets
each subspace specialize.

## References

- Vaswani et al. (2017), §3.2.2.
