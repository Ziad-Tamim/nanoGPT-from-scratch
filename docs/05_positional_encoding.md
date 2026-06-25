# 05 — Positional Encoding

> **Code:** `src/nanogpt/positional_encoding.py` · **Diagram:** `assets/05_positional_encoding.png`
> **Interactive:** [`web/widgets/05_positional_encoding.js`](../web/widgets/05_positional_encoding.js)

## Intuition

Self-attention (doc 06) computes a weighted average over tokens; it has **no notion of
order**. Shuffle the input tokens and the output is just the same shuffle — it sees a *set*,
not a *sequence*. But "dog bites man" ≠ "man bites dog". We fix this by **adding** a vector
that depends on each token's *position* to its embedding, so position 0 and position 5 look
different to the network even when the token is identical.

We add (not concatenate) so the model dimension `C` stays the same and every later layer is
unchanged.

## Math / mechanics — sinusoidal

For position `pos ∈ {0,…,T−1}` and embedding dimension index `i ∈ {0,…,C−1}`, define using
pair index `k = ⌊i/2⌋`:

$$
PE(pos, 2k)   = \sin\!\left(\frac{pos}{\text{base}^{\,2k/C}}\right), \qquad
PE(pos, 2k+1) = \cos\!\left(\frac{pos}{\text{base}^{\,2k/C}}\right),
$$

with `base = 10000`. Even dimensions use `sin`, odd dimensions use `cos`. Each pair `(2k, 2k+1)`
is a sinusoid of angular frequency `ω_k = base^(−2k/C)`. Low dimensions (`k` small) have
`ω ≈ 1` → short wavelength → wiggle fast as position increases; high dimensions have tiny `ω`
→ very long wavelength → wiggle slowly. So a position's encoding is like a multi-resolution
"binary clock" in continuous form. (Drag the dimension slider in the widget to see fast vs.
slow waves; the heatmap shows the full pattern.)

### Why this exact form: relative positions are linear

The key property: the encoding of `pos + δ` is a **fixed linear function** of the encoding of
`pos`, independent of `pos`. For a single frequency `ω_k`, write the 2-D sub-vector
`v_k(pos) = [sin(ω_k·pos), cos(ω_k·pos)]ᵀ`. Then

$$
v_k(pos+\delta) =
\underbrace{\begin{bmatrix} \cos(\omega_k\delta) & \sin(\omega_k\delta) \\
-\sin(\omega_k\delta) & \cos(\omega_k\delta) \end{bmatrix}}_{R(\omega_k\delta)\ \text{(rotation)}}
\; v_k(pos).
$$

This is just the angle-addition identity: shifting position by `δ` **rotates** each frequency
component by a fixed angle `ω_k·δ`. Because the shift→rotation map doesn't depend on `pos`, a
linear attention projection can learn to attend by *relative* offset ("the token 3 back"),
which is exactly what language needs. It also means the scheme extrapolates to sequence
lengths longer than any seen in training.

## Shapes

| Tensor | Shape | Notes |
|---|---|---|
| `pe` buffer | `(T_max, C)` | precomputed once; `T_max = block_size` |
| input `x` | `(B, T, C)` | token embeddings |
| output | `(B, T, C)` | `x + pe[:T]` (broadcasts over batch) |

## Backprop note

Sinusoidal PE is a **constant** (registered as a buffer, not a `Parameter`), so it has zero
gradient — it only shifts activations; the gradient w.r.t. `x` passes through the addition
unchanged (`∂(x+PE)/∂x = 1`). The **learned** variant instead stores a `(block_size, C)`
table that gets gradients exactly like the token embedding (doc 04), updating only the rows
for positions present in the batch.

## Learned variant (what GPT-2 actually uses)

GPT-2 replaces the fixed sinusoids with a trainable `nn.Embedding(block_size, C)` indexed by
position. Pros: can fit dataset-specific position patterns. Cons: cannot generalize beyond
`block_size`, and adds `block_size · C` parameters. Switch via `GPTConfig.pos_encoding`
(`"sinusoidal"` or `"learned"`); `build_positional_encoding` returns the right module.

## References

- Vaswani et al. (2017), *Attention Is All You Need*, §3.5.
- Radford et al. (2019), GPT-2 (learned positional embeddings).
