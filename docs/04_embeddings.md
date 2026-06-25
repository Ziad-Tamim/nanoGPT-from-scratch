# 04 — Token Embeddings

> **Code:** `src/nanogpt/embeddings.py` · **Diagram:** `assets/04_embeddings.png`
> **Status:** stub — fill in during Step 3.

## Intuition

Each token id indexes a row of a learnable table, turning a discrete id into a dense
`C`-dimensional vector the network can manipulate. Similar tokens can learn similar vectors.

## Math / mechanics

- Embedding table `E ∈ ℝ^{vocab_size × C}` (learnable).
- Lookup: for token id `i`, embedding `= E[i]`. For a batch this is a gather, not a matmul.
- Equivalent to one-hot `(B, T, vocab_size)` × `E` — but implemented as indexing.

## Shapes

- Input ids `(B, T)` → output `(B, T, C)`.

## Backprop note

Gradient flows only to the rows that were looked up (sparse update). `∂L/∂E[i]` accumulates
over every position where token `i` appeared in the batch.

## References

- Vaswani et al. (2017), §3.4 (embeddings & softmax weight sharing).
