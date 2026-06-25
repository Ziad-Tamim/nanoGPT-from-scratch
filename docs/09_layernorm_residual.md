# 09 — Layer Normalization & Residual Connections

> **Code:** `src/nanogpt/layer_norm.py` · **Diagram:** `assets/09_layernorm_residual.png`

Two small ideas that together make deep Transformers trainable.

## Residual connections

### Intuition

Instead of replacing the input, each sublayer *adds* to it:

$$x \leftarrow x + \mathrm{Sublayer}(x).$$

This gives the signal (and the gradient) a direct "skip" path around the sublayer. The network
only has to learn a **residual** — a correction to `x` — which is easier than relearning the
whole representation, and it keeps very deep stacks from degrading.

### Why it fixes vanishing gradients

The Jacobian of a residual block is

$$\frac{\partial(x + \mathrm{Sublayer}(x))}{\partial x} = I + \frac{\partial\,\mathrm{Sublayer}}{\partial x}.$$

The `+I` term means that during backprop the upstream gradient is passed through **unattenuated**
no matter what the sublayer does. Across `N` stacked blocks the gradient is a product of
`(I + J_k)` terms rather than a product of small `J_k` terms, so it cannot vanish the way it
does in a plain deep network (or an RNN — recall the RNN problems in the background diagram).

## Layer normalization

### Intuition

As signals flow through many layers, the scale of activations can drift, slowing or
destabilizing training. LayerNorm renormalizes **each token's** feature vector to zero mean
and unit variance, then lets the model rescale with learnable parameters.

### Math

For a single token vector `x ∈ ℝ^C`, over the feature axis:

$$
\mu = \frac{1}{C}\sum_{j=1}^{C} x_j, \qquad
\sigma^2 = \frac{1}{C}\sum_{j=1}^{C}(x_j - \mu)^2,
$$
$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
\mathrm{LN}(x) = \gamma \odot \hat{x} + \beta.
$$

`γ, β ∈ ℝ^C` are learnable (initialized to 1 and 0). `ε` (default `1e-5`) guards against
division by zero. Note `σ²` uses the **population** variance (divide by `C`, not `C−1`) — this
matches `torch.nn.LayerNorm`, verified in `test_matches_pytorch_layernorm`.

Unlike BatchNorm, statistics are per-token and never depend on the batch, so LayerNorm behaves
identically in training and inference and works at any batch size.

## Pre-LN vs post-LN

Where you put the LayerNorm matters:

| Variant | Formula | Used by |
|---|---|---|
| **Post-LN** | `x = LN(x + Sublayer(x))` | original Transformer (2017) |
| **Pre-LN** | `x = x + Sublayer(LN(x))` | GPT-2, and this project |

**Pre-LN** normalizes the *input* to each sublayer but leaves the residual path itself
un-normalized — so the clean `+I` gradient highway above runs unbroken from the output all the
way to the embeddings. This makes deep models far more stable to train (often without learning-
rate warmup). Our Transformer block (doc 10) uses pre-LN.

## Backprop note

LayerNorm's backward pass is more involved than it looks because `μ` and `σ²` both depend on
every `x_j`, coupling the gradients across the feature axis:

$$
\frac{\partial L}{\partial x_i} =
\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\left(
\frac{\partial L}{\partial \hat{x}_i}
- \frac{1}{C}\sum_j \frac{\partial L}{\partial \hat{x}_j}
- \frac{\hat{x}_i}{C}\sum_j \frac{\partial L}{\partial \hat{x}_j}\,\hat{x}_j
\right).
$$

The two subtracted terms remove the components of the incoming gradient along the mean and
along `x̂` directions (autograd handles this for us; it's shown here for understanding).

## References

- Ba, Kiros & Hinton (2016), *Layer Normalization*.
- He et al. (2016), *Deep Residual Learning* (residual connections).
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* (pre-LN).
