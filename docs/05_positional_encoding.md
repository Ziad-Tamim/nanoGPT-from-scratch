# 05 — Positional Encoding

> **Code:** `src/nanogpt/positional_encoding.py` · **Diagram:** `assets/05_positional_encoding.png`
> **Status:** stub — fill in during Step 4.

## Intuition

Self-attention is **permutation-invariant** — it sees a *set* of tokens, not a sequence.
Positional encoding injects "where am I" so order matters. We add a position vector to each
token embedding.

## Math / mechanics — sinusoidal (the diagram's version)

For position `pos` and dimension `i` (0 ≤ i < C):

```
PE(pos, 2k)   = sin( pos / 10000^(2k / C) )
PE(pos, 2k+1) = cos( pos / 10000^(2k / C) )
```

- Even dims use sin, odd dims use cos; wavelengths form a geometric progression from 2π to
  ~10000·2π. Each dimension is a sinusoid of a different frequency.
- Key property: `PE(pos+δ)` is a **linear function** of `PE(pos)`, so the model can attend by
  *relative* offsets. (Show the rotation-matrix derivation here.)
- Added, not concatenated: `x = token_embedding + PE`.

## Learned variant (what GPT actually uses)

A second `nn.Embedding(block_size, C)` indexed by position; trained like any parameter.
Config switches between sinusoidal and learned.

## Shapes

- `PE` is `(T, C)`, broadcast-added to `(B, T, C)`.

## Backprop note

Sinusoidal PE is **fixed** (no gradient). Learned PE gets gradients like the token embedding.

## References

- Vaswani et al. (2017), §3.5.
