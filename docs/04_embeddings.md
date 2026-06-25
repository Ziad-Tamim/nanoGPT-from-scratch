# 04 — Token Embeddings

> **Code:** `src/nanogpt/embeddings.py` · **Diagram:** `assets/04_embeddings.png`

## Intuition

A token id like `42` is just a name — it carries no meaning a network can use (id 42 is not
"twice as much" as id 21). The embedding layer replaces each id with a **learnable vector** of
length `C = n_embd`. During training these vectors arrange themselves so that tokens used in
similar ways end up with similar vectors. This is the first place the model *learns* anything.

## Math / mechanics

The embedding is a table (matrix) of learnable parameters:

$$E \in \mathbb{R}^{V \times C}, \qquad V = \texttt{vocab\_size},\ C = \texttt{n\_embd}.$$

Row `i` of `E`, written `E[i] ∈ ℝ^C`, is the embedding of token id `i`. For a token id `t`:

$$\text{embed}(t) = E[t].$$

### Lookup = one-hot matmul

Indexing is mathematically identical to multiplying a one-hot row vector by `E`. If `e_t` is
the one-hot vector for token `t` (a `V`-vector that is 1 at index `t`, else 0):

$$e_t^\top E = E[t].$$

So the embedding layer is a linear layer with **no bias** whose input is always one-hot. We
implement it as indexing because that is far cheaper than building `V`-wide one-hot tensors
and multiplying — but the equivalence is why gradients behave the way they do (below). This is
verified directly in `tests/test_embeddings.py::test_equivalent_to_onehot_matmul`.

## Shapes

| Tensor | Shape | Notes |
|---|---|---|
| input ids | `(B, T)` | integers in `[0, V)` |
| table `E` | `(V, C)` | learnable parameters |
| output | `(B, T, C)` | one `C`-vector per token |

Every later component preserves the trailing `C`, so from here on the data is `(B, T, C)`.

## Backprop note

Because the effective input is one-hot, the gradient w.r.t. the table is **sparse**: only the
rows for ids that appeared in the batch receive gradient. Concretely, if the loss is `L` and a
position used token `t` with output gradient `g ∈ ℝ^C`, then

$$\frac{\partial L}{\partial E[t]} \mathrel{+}= g,$$

accumulated over every position in the batch that used token `t`. Rows for unused ids get
exactly zero gradient (tested in `test_gradient_is_sparse_to_used_rows`). This is why
embeddings for rare tokens train slowly — they simply receive fewer updates.

## Initialization

We initialize `E ~ N(0, 0.02²)` following GPT-2. Small initial weights keep early activations
— and therefore the softmax logits at the output — in a numerically stable range before the
model has learned anything.

## Position information

Token embeddings alone are **order-blind**: the same token gets the same vector regardless of
where it sits in the sequence. Order is injected next, by adding positional encodings — see
[05_positional_encoding.md](05_positional_encoding.md).

## References

- Vaswani et al. (2017), *Attention Is All You Need*, §3.4.
- Radford et al. (2019), GPT-2 (initialization scheme).
