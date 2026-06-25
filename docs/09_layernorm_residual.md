# 09 — Layer Normalization & Residual Connections

> **Code:** `src/nanogpt/layer_norm.py` · **Diagram:** `assets/09_layernorm_residual.png`
> **Status:** stub — fill in during Step 8.

## Intuition

**Residuals** give gradients a direct highway around each sublayer, so deep stacks train.
**LayerNorm** keeps each token's activation distribution stable (zero mean, unit variance)
so training doesn't blow up or stall.

## Math / mechanics

LayerNorm over the feature dimension `C`, per token:

```
μ  = mean(x, axis=C)          σ² = var(x, axis=C)
x̂ = (x − μ) / √(σ² + ε)
LN(x) = γ ⊙ x̂ + β             # γ, β learnable, shape (C,)
```

Residual: `x = x + Sublayer(x)`.

### Pre-LN vs post-LN

- **Post-LN** (original paper): `x = LN(x + Sublayer(x))`.
- **Pre-LN** (GPT-2, what we use): `x = x + Sublayer(LN(x))`. The residual path stays clean
  (un-normalized), which makes deep models far more stable to train. Explain the gradient-flow
  argument here.

## Shapes

LayerNorm and residual are both `(B, T, C)` → `(B, T, C)` (shape-preserving).

## Backprop note

Residual adds an identity term to the Jacobian (`∂out/∂x = I + ∂Sublayer/∂x`), preventing
vanishing gradients. Include the LayerNorm backward formula (depends on `x̂`, `γ`, and the
mean/var statistics).

## References

- Ba et al. (2016), Layer Normalization; Xiong et al. (2020), On Layer Normalization (pre-LN).
