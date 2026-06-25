# 06 — Self-Attention (single head)

> **Code:** `src/nanogpt/attention.py` · **Diagram:** `assets/06_self_attention.png`
> **This is the core mechanism of the Transformer.**

## Intuition

Each token needs to gather information from other tokens. Attention does this with a
soft, learned "database lookup":

- Every token forms a **query** `q` — *what am I looking for?*
- Every token also exposes a **key** `k` — *what do I contain?* — and a **value** `v` —
  *what will I hand over if you attend to me?*
- Token `i` compares its query to every key, turns the similarities into weights that sum to
  1, and reads out the weighted average of the corresponding values.

So tokens **communicate**: a verb can pull information from its subject, a closing quote from
its opening quote, etc. The "self" means queries, keys, and values all come from the same
sequence.

## Math / mechanics

From input `x ∈ ℝ^{B×T×C}`, three learned linear maps (no bias) produce queries, keys, values
of width `d = head_size`:

$$Q = xW_Q,\quad K = xW_K,\quad V = xW_V \qquad (W_\bullet \in \mathbb{R}^{C\times d}).$$

Scaled dot-product attention:

$$
\text{scores} = \frac{QK^\top}{\sqrt{d}}\in\mathbb{R}^{B\times T\times T},\qquad
A = \mathrm{softmax}(\text{scores}),\qquad
\text{out} = A\,V.
$$

`scores[b, i, j]` is how much token `i` (its query) matches token `j` (its key). `softmax`
is taken over the **last** axis (`j`), so each query's weights over keys sum to 1.

### The causal mask

For a language model, token `i` must not see tokens `j > i` (the future it's predicting). We
set those scores to `−∞` **before** softmax, which drives their weight to exactly 0:

$$
\text{scores}[i, j] \leftarrow -\infty \quad\text{for } j > i.
$$

Implemented with a lower-triangular `tril` mask. Result: `A` is lower-triangular, row `i`'s
weight is spread only over positions `0..i`. The test
`test_future_tokens_do_not_affect_past_outputs` confirms perturbing a later token leaves
earlier outputs unchanged.

### Why divide by √d

Let query and key components be independent with mean 0 and variance 1. Their dot product
`q·k = Σ_{m=1}^{d} q_m k_m` is a sum of `d` zero-mean, unit-variance terms, so

$$\operatorname{Var}(q\cdot k) = d, \qquad \text{std} = \sqrt{d}.$$

Without scaling, scores grow like `√d`; for large `d` the softmax sees very large/small
inputs, saturates (one weight ≈ 1, rest ≈ 0), and its gradient nearly vanishes. Dividing by
`√d` normalizes the variance back to ≈ 1, keeping softmax in its responsive, well-gradient
region. (Vaswani et al. §3.2.1.)

## Shapes

| Tensor | Shape |
|---|---|
| `x` | `(B, T, C)` |
| `Q, K, V` | `(B, T, d)` with `d = head_size` |
| `scores`, `A` (weights) | `(B, T, T)` |
| `out` | `(B, T, d)` |

## Backprop note

**Softmax Jacobian.** For one row `a = softmax(s)`,

$$\frac{\partial a_i}{\partial s_j} = a_i(\delta_{ij} - a_j).$$

Masked entries have `a_j = 0` and receive zero gradient — the `−∞` positions are inert in both
passes. The `1/√d` factor scales the upstream gradient into `Q, K`, which is exactly why it
keeps gradients healthy (see the variance argument above). Gradients flow into `W_Q, W_K, W_V`
(tested in `test_gradients_flow_to_projections`).

## From one head to many

A single head learns one kind of relationship. The next component runs several heads in
parallel in smaller subspaces and recombines them — see
[07_multi_head_attention.md](07_multi_head_attention.md).

## References

- Vaswani et al. (2017), §3.2.1 (Scaled Dot-Product Attention).
- Karpathy, *Let's build GPT* (the single-head derivation).
