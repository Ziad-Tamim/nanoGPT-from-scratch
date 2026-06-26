# Architecture & File Structure

## Decoder-only GPT {#decoder-only}

The original *Attention Is All You Need* diagram (see
[`assets/01_transformer_architecture.png`](assets/01_transformer_architecture.png)) shows an
**encoder–decoder** model built for translation. GPT keeps only the **right-hand
(decoder) stack**, and drops the cross-attention that pointed at the encoder. What remains:

```
tokens ─▶ token embedding ─(+)─ positional encoding ─▶ [ Transformer block ] × N ─▶ final LayerNorm ─▶ Linear (LM head) ─▶ softmax ─▶ next-token probabilities
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          │  pre-LN → Masked Multi-Head Attention   │  (+ residual)
                                          │  pre-LN → Feed-Forward (MLP)            │  (+ residual)
                                          └─────────────────────────────────────────┘
```

"Masked" / causal means a token at position *t* can only attend to positions ≤ *t* — so the
model never sees the future it's trying to predict. That single masking choice is what turns
the encoder block into a left-to-right language model.

### Why this over an RNN (the background diagram)

The RNN panel lists three problems: slow sequential computation, vanishing/exploding
gradients, and trouble using information from far back. Self-attention fixes all three: every
position attends to every other position **in parallel** (no sequential dependency), the path
length between any two tokens is O(1) (so gradients flow directly), and long-range
dependencies are a single attention lookup away.

## Data flow & tensor shapes

Shapes use **(B, T, C)** = (batch, time/sequence length, channels = `n_embd`).

| Stage | Module | In → Out |
|---|---|---|
| Token ids | `data.py` | `(B, T)` ints |
| Token embedding | `embeddings.py` | `(B, T)` → `(B, T, C)` |
| + Positional encoding | `positional_encoding.py` | `(B, T, C)` → `(B, T, C)` |
| × N Transformer blocks | `block.py` | `(B, T, C)` → `(B, T, C)` |
| Final LayerNorm | `layer_norm.py` | `(B, T, C)` → `(B, T, C)` |
| LM head (Linear) | `model.py` | `(B, T, C)` → `(B, T, vocab_size)` |
| Loss | `model.py` | cross-entropy vs targets `(B, T)` |

Inside one block, attention splits C into `n_head` heads of size `C / n_head`, runs
scaled dot-product attention per head, concatenates back to C, and projects.

## File structure

```
.
├── README.md                       # GitHub landing page
├── pyproject.toml                  # deps (add torch w/ CUDA)
├── .gitignore                      # ignores checkpoints/, data/
├── docs/                           # all explanations + diagrams (this folder)
│   ├── README.md                   # docs index
│   ├── 00_plan.md                  # roadmap & steps
│   ├── 01_architecture.md          # this file
│   ├── 02_dataset.md
│   ├── 03_tokenization.md … 13_sampling.md
│   └── assets/                     # exported Excalidraw images
│       ├── README.md               # naming convention + checklist
│       └── *.png
├── scripts/
│   └── prepare_data.py             # download Tiny Shakespeare
├── src/
│   └── nanogpt/
│       ├── __init__.py
│       ├── config.py               # GPTConfig dataclass + device select
│       ├── tokenizer.py            # char-level encode/decode
│       ├── data.py                 # batching, train/val split
│       ├── embeddings.py           # token embedding
│       ├── positional_encoding.py  # sinusoidal (+ learned option)
│       ├── attention.py            # single-head causal attention
│       ├── multi_head_attention.py # multi-head + projection
│       ├── feed_forward.py         # position-wise MLP
│       ├── layer_norm.py           # hand-written LayerNorm
│       ├── block.py                # one Transformer block (Nx unit)
│       ├── model.py                # full GPT: forward + loss + generate
│       ├── train.py                # training loop + checkpoints
│       └── generate.py             # load checkpoint, sample text
├── tests/                          # tiny shape/sanity tests per component
├── data/                           # downloaded corpus (gitignored)
└── checkpoints/                    # saved models (gitignored)
```

## Design principles

- **One concept per file.** If a file needs two paragraphs to explain "what it is," it's
  probably two files.
- **Every module references its diagram and doc** in its top docstring, e.g.
  `"""... See docs/06_self_attention.md and docs/assets/06_self_attention.png."""`
- **Shapes are asserted/commented** at boundaries so the (B, T, C) story is always visible.
- **Config-driven.** All hyperparameters live in `GPTConfig`; no magic numbers in modules.
