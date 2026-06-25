# Build Plan & Roadmap

The plan for building a decoder-only GPT from scratch to understand the Transformer.

## Goal

Train a character-level GPT on Tiny Shakespeare and generate text — while writing **every
component by hand** so the math is fully understood. Each component is its own small,
documented file with a matching `docs/` page and Excalidraw diagram.

## Assumptions & decisions

These were confirmed up front. If any change, revisit the affected step.

1. **Framework:** PyTorch for tensors + autograd, but components (attention, positional
   encoding, layer norm, feed-forward, the block, the model) are implemented manually.
   We do **not** use `torch.nn.Transformer` / `MultiheadAttention`. We *do* use
   `nn.Linear`, `nn.Embedding`, autograd, and the optimizer (re-implementing backprop by
   hand is out of scope; the math of backprop is *explained* in the docs instead).
2. **Architecture:** Decoder-only (GPT). No encoder, no cross-attention. Causal
   (masked) self-attention only.
3. **Dataset:** Tiny Shakespeare (~1 MB plain text), downloaded by a script.
4. **Tokenizer:** Character-level. Vocab ≈ 65 unique characters. Code is structured so a
   BPE tokenizer can be dropped in later.
5. **Positional encoding:** We implement **sinusoidal** encoding (matches the diagram and
   is great for the math), and also show the **learned** embedding variant GPT actually
   uses. Config picks which one.
6. **Hardware:** NVIDIA GPU (CUDA) with automatic CPU fallback. Default config is small
   enough to train in minutes.
7. **Math depth:** Rigorous — formulas, tensor shapes at each step, gradient/backprop
   notes, references to the original papers.
8. **Diagrams:** Exported PNG/SVG from Excalidraw into `docs/assets/`, one per component,
   referenced from both the doc page and the module docstring.

## Default model config (starting point)

A small config that trains quickly on a single GPU and still produces readable text.

| Hyperparameter | Symbol | Default | Note |
|---|---|---|---|
| Vocab size | `vocab_size` | ~65 | Set from the dataset |
| Context length | `block_size` (T) | 256 | Max tokens of context |
| Embedding dim | `n_embd` (C) | 384 | Model width |
| Heads | `n_head` | 6 | `head_size = n_embd / n_head = 64` |
| Layers | `n_layer` | 6 | Transformer blocks |
| Dropout | `dropout` | 0.2 | Regularization |
| Batch size | `batch_size` (B) | 64 | Sequences per step |
| Learning rate | `lr` | 3e-4 | AdamW |

Tensor shape convention used throughout the docs: **(B, T, C)** = (batch, time/sequence,
channels/embedding).

## Build order (steps)

Build bottom-up; each step has working code, a test, a doc page, and a diagram. Check off
as you go.

- [x] **Step 0 — Project setup.** Add PyTorch (CUDA) to `pyproject.toml`, create
  `src/nanogpt/` package, `config.py` with a `GPTConfig` dataclass, device auto-detect.
  ✅ torch 2.6.0+cu124 on RTX 4070 (sm_89); 5/5 config tests pass.
- [x] **Step 1 — Data pipeline.** `scripts/prepare_data.py` downloads Tiny Shakespeare;
  `data.py` does train/val split and random batch sampling. → doc 02
  ✅ 1.1 M chars downloaded; `TextData.get_batch` verified (y = x shifted by 1) on GPU.
- [x] **Step 2 — Tokenizer.** `tokenizer.py`: build vocab (65 chars), `encode`/`decode`,
  save/load. → doc 03 ✅ lossless round-trip tested.
- [x] **Step 3 — Embeddings.** `embeddings.py`: token embedding table (B,T)→(B,T,C). → doc 04
  ✅ shape/one-hot-equivalence/sparse-gradient tests pass; doc 04 written.
- [x] **Step 4 — Positional encoding.** `positional_encoding.py`: sinusoidal (+ learned
  option), the math of why sin/cos. → doc 05
  ✅ formula-match + no-params + factory tests pass; doc 05 (incl. rotation/relative-pos
  derivation) written; matches the web widget.
- [x] **Step 5 — Self-attention (single head).** `attention.py`: Q,K,V, scaled dot-product,
  causal mask, softmax. The heart of the model. → doc 06
  ✅ causality, row-sum-1, mask, gradient tests pass; doc 06 (√d variance + softmax Jacobian)
  written. `Head` can return weights for the upcoming attention widget.
- [x] **Step 6 — Multi-head attention.** `multi_head_attention.py`: parallel heads, concat,
  output projection. → doc 07
  ✅ shape-preserved, head-count/size, per-head causality, indivisible-dims tests pass; doc 07
  written (incl. why splitting costs no extra params).
- [x] **Step 7 — Feed-forward network.** `feed_forward.py`: position-wise MLP (4× expansion,
  GELU). → doc 08
  ✅ shape, 4x-hidden, position-wise, non-linearity, gradient tests pass; doc 08 written.
- [x] **Step 8 — LayerNorm + residuals.** `layer_norm.py`: hand-written LayerNorm; explain
  pre-LN vs post-LN and why residual connections help gradients. → doc 09
  ✅ matches torch.nn.LayerNorm; zero-mean/unit-var + gradient tests pass; doc 09 (residual
  +I highway, LN backward formula) written.
- [x] **Step 9 — Transformer block.** `block.py`: assemble attention + FFN with pre-LN and
  residuals (the repeated Nx unit). → doc 10
  ✅ shape, causality-through-block, all-subcomponent gradients, residual-identity tests pass;
  doc 10 written.
- [x] **Step 10 — GPT model.** `model.py`: embeddings + positions + N blocks + final
  LayerNorm + LM head; the forward pass and the loss (cross-entropy). → doc 11
  ✅ 10.67 M params; init loss 4.30 ≈ ln(65); weight tying, generate, return-weights tests
  pass; live GPU forward + untrained sample verified; doc 11 written.
- [x] **Step 11 — Training loop.** `train.py`: `GPTConfig`, AdamW, train/val loss eval,
  checkpoint save/load, logging. → doc 12
  ✅ decay-group split, estimate_loss, loss-decreases, checkpoint-loadable tests pass;
  doc 12 written. Run: `uv run python -m nanogpt.train`.
- [x] **Step 12 — Sampling / generation.** `generate.py`: autoregressive sampling with
  temperature and top-k; load a checkpoint and print text. → doc 13
  ✅ checkpoint load, prompt-prefix, in-vocab, temp/top-k variants tests pass; doc 13 written.
  Run: `uv run python -m nanogpt.generate --prompt "ROMEO:" --tokens 500`.
- [ ] **Step 13 — Polish.** Loss-curve plot (matplotlib/seaborn already installed), README
  results section, final pass over docs and diagrams.

## Definition of done per component

A component is "done" when:
1. The module exists and is imported into the model.
2. A tiny shape test passes (input shape → expected output shape).
3. Its `docs/NN_*.md` page is written (intuition, math, shapes, backprop note, references).
4. Its diagram exists in `docs/assets/` and is linked from both the doc and the docstring.

## Diagrams to draw (Excalidraw)

See [assets/README.md](assets/README.md) for the naming convention and the full checklist.

## Parallel track — interactive web visualizer

Alongside the model, an interactive site in `web/` (vanilla JS + D3, static, GitHub Pages)
gives each component a live widget. Build a widget when its component is done. The
positional-encoding widget is already built as the proof of concept. See
[14_web_visualizer.md](14_web_visualizer.md).
