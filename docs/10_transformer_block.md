# 10 — Transformer Block (the Nx unit)

> **Code:** `src/nanogpt/block.py` · **Diagram:** `assets/10_transformer_block.png`

## Intuition

This is the unit repeated `n_layer` times — the `Nx` box in the architecture diagram. It
combines the two complementary operations we've built:

1. **Communication** — multi-head self-attention (doc 07): tokens look at each other and
   exchange information.
2. **Computation** — the position-wise feed-forward network (doc 08): each token independently
   transforms what it gathered.

Each is wrapped in **pre-LayerNorm** (doc 09) and a **residual** connection. Stacking blocks
lets the model alternate "gather context" and "process it" many times, building increasingly
abstract representations with depth.

## Math / mechanics (pre-LN)

$$
\begin{aligned}
x &\leftarrow x + \mathrm{MHA}\big(\mathrm{LN}_1(x)\big) \\
x &\leftarrow x + \mathrm{FFN}\big(\mathrm{LN}_2(x)\big)
\end{aligned}
$$

Two sublayers, two LayerNorms, two residual adds. Crucially the LayerNorm is applied to the
input *of* each sublayer, while the residual `x` flows through untouched — so the gradient
highway from doc 09 runs cleanly through every block.

The block is **shape-preserving**: `(B, T, C) → (B, T, C)`. That invariance is what lets us
stack an arbitrary number of them.

## Why attention first, then FFN

Attention enriches each token with context from the rest of the sequence; the FFN then does
the heavier non-linear processing on that enriched vector. The order matters less than the
fact that both sit on residual branches, so neither can disrupt the identity path (verified in
`test_residual_path_present`: zero both sublayer outputs and the block becomes the identity).

## Shapes

| Tensor | Shape |
|---|---|
| input / output | `(B, T, C)` |
| attention weights (viz) | `(B, n_head, T, T)` |

## Backprop note

Because of the two residual adds, the block's input gradient is

$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial x_{\text{out}}}\Big( I + \tfrac{\partial \mathrm{FFN}\circ\mathrm{LN}_2}{\partial \cdot}\Big)\Big( I + \tfrac{\partial \mathrm{MHA}\circ\mathrm{LN}_1}{\partial \cdot}\Big),
$$

i.e. an `I + (small)` factor per sublayer. Stacked over `n_layer` blocks this is a product of
near-identity terms, so gradients neither vanish nor explode through depth. Causality is
preserved by the whole block (`test_causality_survives_full_block`).

## References

- Vaswani et al. (2017), Figure 1.
- Radford et al. (2019), GPT-2 (pre-LN decoder block).
