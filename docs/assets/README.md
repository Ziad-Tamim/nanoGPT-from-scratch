# Diagrams (Excalidraw exports)

Hand-drawn diagrams that explain each component. Export from Excalidraw as **PNG** (for
GitHub rendering) — optionally also keep the editable **`.excalidraw`** source next to it.

## Naming convention

`NN_component_name.png` where `NN` matches the doc number, e.g. `05_positional_encoding.png`
is referenced by `docs/05_positional_encoding.md` and by the module docstring in
`src/nanogpt/positional_encoding.py`.

Keep a matching `NN_component_name.excalidraw` source if you want it editable later.

## How to reference a diagram

- **In a doc page (Markdown):**
  `![Positional encoding](assets/05_positional_encoding.png)`
- **In a code module (docstring):**
  `See docs/05_positional_encoding.md and docs/assets/05_positional_encoding.png`

## Checklist

- [ ] `00_rnn.png` — background: the RNN and its problems (slow, vanishing gradients, long-range)
- [ ] `01_transformer_architecture.png` — the full Transformer architecture (the encoder–decoder figure / decoder-only stack)
- [ ] `02_dataset_batching.png` — sliding window: inputs `x` and shifted targets `y`
- [ ] `03_tokenization.png` — char ↔ int mapping (stoi / itos)
- [ ] `04_embeddings.png` — token id → embedding vector (lookup table)
- [ ] `05_positional_encoding.png` — sin/cos curves across position & dimension
- [ ] `06_self_attention.png` — Q,K,V, scaled dot-product, causal mask, softmax
- [ ] `07_multi_head_attention.png` — split into heads → concat → project
- [ ] `08_feed_forward.png` — position-wise MLP (expand → GELU → project)
- [ ] `09_layernorm_residual.png` — LayerNorm + residual/skip connection
- [ ] `10_transformer_block.png` — full block (pre-LN, attention, FFN, residuals)
- [ ] `11_gpt_model.png` — end-to-end stack: embeddings → N blocks → LM head
- [ ] `12_training.png` — forward → cross-entropy loss → backward → optimizer step
- [ ] `13_sampling.png` — autoregressive generation loop (temperature / top-k)

> Tip: keep one master `.excalidraw` file with all panels, then export each frame to its
> own PNG using Excalidraw's per-frame export.
