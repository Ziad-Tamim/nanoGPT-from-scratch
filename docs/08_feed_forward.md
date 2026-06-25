# 08 — Feed-Forward Network (MLP)

> **Code:** `src/nanogpt/feed_forward.py` · **Diagram:** `assets/08_feed_forward.png`

## Intuition

Attention is mostly a **weighted average** of value vectors — a fundamentally *linear* mixing
operation (the only non-linearity is the softmax that produces the weights). On its own it
can't compute rich per-token functions. The feed-forward network adds that capacity: after
each token has gathered context via attention, the FFN lets it **think** — a small non-linear
MLP applied independently at every position.

A useful mental model: attention decides *what information to fetch*; the FFN decides *what to
do with it*.

## Math / mechanics

A two-layer MLP with a widened hidden layer:

$$
\mathrm{FFN}(x) = \mathrm{Dropout}\big( W_2\,\phi(W_1 x + b_1) + b_2 \big),
$$

with `W_1 ∈ ℝ^{C×4C}`, `W_2 ∈ ℝ^{4C×C}`, and `φ = GELU`. Applied to `(B, T, C)`, the linear
layers act on the last axis only, so each of the `B·T` token vectors is transformed by the
**same** function with **no interaction across positions** — that's what "position-wise"
means (verified in `test_is_position_wise`).

### Why 4× expansion

The hidden width is `4C` by convention (Vaswani et al.). Projecting up to a higher-dimensional
space gives the non-linearity room to carve more complex decision boundaries before
compressing back to `C`. This block holds a large share of the model's parameters: roughly
`2 · C · 4C = 8C²` weights per layer — typically about ⅔ of a Transformer block's parameters.

### GELU vs ReLU

We use **GELU** (Gaussian Error Linear Unit), as in GPT-2:

$$
\mathrm{GELU}(z) = z\,\Phi(z),
$$

where `Φ` is the standard-normal CDF. Unlike ReLU's hard cutoff at 0, GELU is smooth and
gently lets small negative values through, which empirically trains better in Transformers.
The original Transformer used ReLU (`max(0, z)`); swapping it is a one-line change.

## Shapes

| Stage | Shape |
|---|---|
| input | `(B, T, C)` |
| after `W_1` (expand) | `(B, T, 4C)` |
| after `W_2` (project) | `(B, T, C)` |

## Backprop note

Standard MLP gradients: `∂L/∂W_2`, `∂L/∂W_1` via the chain rule, with the GELU derivative
`Φ(z) + z·φ(z)` (where `φ` is the normal PDF) in the middle. Because the layer is
position-wise, gradients for different positions are independent and simply accumulate into
the shared weights.

## References

- Vaswani et al. (2017), §3.3 (Position-wise Feed-Forward Networks).
- Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)*.
