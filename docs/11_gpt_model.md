# 11 — The GPT Model (assembly)

> **Code:** `src/nanogpt/model.py` · **Diagram:** `assets/11_gpt_model.png`

## Intuition

Everything we built snaps together here into one language model: embed the tokens, add their
positions, run them through a stack of Transformer blocks, normalize, and project to a
probability distribution over the vocabulary at every position. The forward pass also computes
the training loss, and a `generate` method lets the trained model write text.

## Math / mechanics

$$
\begin{aligned}
h_0 &= \mathrm{Dropout}\big(\mathrm{TokenEmb}(idx) + \mathrm{PosEnc}\big) &&(B, T, C)\\
h_\ell &= \mathrm{Block}_\ell(h_{\ell-1}), \quad \ell = 1\ldots N &&(B, T, C)\\
z &= \mathrm{LN}_f(h_N) &&(B, T, C)\\
\text{logits} &= z\,W_{\text{head}} &&(B, T, V)
\end{aligned}
$$

### Loss

Next-token prediction is multi-class classification over the vocabulary at every position.
We flatten the batch and time axes and apply cross-entropy:

$$
L = -\frac{1}{B\,T}\sum_{b,t} \log p_\theta\big(\text{target}_{b,t}\mid x_{b,\le t}\big),
\qquad p = \mathrm{softmax}(\text{logits}).
$$

**Sanity check at initialization.** With small random weights the logits are ≈ 0, so the
softmax is ≈ uniform over `V` tokens and the loss should be ≈ `ln(V)`. For Shakespeare
`V = 65`, that's `ln 65 ≈ 4.17` — and the model measures ~4.3 at init
(`test_initial_loss_near_ln_vocab`). If your init loss is wildly off, something is miswired.

### Weight tying

The input embedding `E ∈ ℝ^{V×C}` and the output head `W_head ∈ ℝ^{C×V}` both encode the same
token↔vector correspondence, so we **share** them (`lm_head.weight = token_embedding.weight`).
This saves `V·C` parameters and typically improves quality (Press & Wolf, 2017). Verified by
`test_weight_tying_shares_storage` (same underlying tensor).

## Shapes

| Tensor | Shape |
|---|---|
| `idx` | `(B, T)` |
| hidden states | `(B, T, C)` |
| `logits` | `(B, T, V)` |
| `targets` | `(B, T)` |
| `loss` | scalar |
| attention weights (viz) | `(B, n_layer, n_head, T, T)` |

## Generation

`generate(idx, max_new_tokens, temperature, top_k)` samples autoregressively: predict the
next-token distribution, sample one token, append, repeat — cropping the context to the last
`block_size` tokens each step. Details and the sampling knobs are in
[13_sampling.md](13_sampling.md).

## Backprop note

Cross-entropy over softmax has the clean gradient

$$\frac{\partial L}{\partial \text{logits}_{b,t}} = \frac{1}{BT}\big(\mathrm{softmax}(\text{logits}_{b,t}) - \text{onehot}(\text{target}_{b,t})\big),$$

which flows back through the LM head, the final LayerNorm, every block (each preserving it via
residuals), and finally into the embeddings. Because the head weight is tied to the embedding,
that tensor receives gradient from **both** the input lookup and the output projection.

## Default size

The default config (`n_embd=384, n_head=6, n_layer=6, block_size=256`) is about **10.7 M
parameters** — small enough to train in minutes on a single GPU.

## References

- Radford et al. (2018/2019), GPT / GPT-2.
- Press & Wolf (2017), *Using the Output Embedding to Improve Language Models* (weight tying).
