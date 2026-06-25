# 10 — Transformer Block (the Nx unit)

> **Code:** `src/nanogpt/block.py` · **Diagram:** `assets/10_transformer_block.png`
> **Status:** stub — fill in during Step 9.

## Intuition

The repeated unit (the `Nx` box in the diagram). It interleaves **communication**
(attention — tokens talk) and **computation** (feed-forward — each token thinks), each wrapped
in pre-LN + residual. Stacking `n_layer` of these builds depth.

## Math / mechanics (pre-LN)

```
x = x + MultiHeadAttention( LayerNorm(x) )
x = x + FeedForward(        LayerNorm(x) )
```

Two sublayers, two LayerNorms, two residual adds. Shape preserved end to end.

## Shapes

`(B, T, C)` → `(B, T, C)`.

## Why this order

Attention first lets each token gather context; the FFN then transforms the enriched
representation. Both on residual branches so the identity path is never disturbed (see doc 09).

## References

- Vaswani et al. (2017), Figure 1; Radford et al. (2019), GPT-2.
