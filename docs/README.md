# Documentation Index

This folder explains the project and the math behind every component of the Transformer.
Read it top to bottom to learn, or jump to a component.

## Planning

| Doc | What's inside |
|---|---|
| [00_plan.md](00_plan.md) | Build roadmap, assumptions, step-by-step order, progress checklist |
| [01_architecture.md](01_architecture.md) | File/folder structure and how data flows through the model |
| [02_dataset.md](02_dataset.md) | Tiny Shakespeare: download, char tokenization, batching, train/val split |

## Components (one file per Transformer piece)

Each page follows the same template: **intuition → math (with shapes) → backprop notes →
code reference → diagram**. Diagrams live in [`assets/`](assets/).

| # | Doc | Code module | Diagram |
|---|---|---|---|
| 03 | [03_tokenization.md](03_tokenization.md) | `src/nanogpt/tokenizer.py` | `assets/03_tokenization.png` |
| 04 | [04_embeddings.md](04_embeddings.md) | `src/nanogpt/embeddings.py` | `assets/04_embeddings.png` |
| 05 | [05_positional_encoding.md](05_positional_encoding.md) | `src/nanogpt/positional_encoding.py` | `assets/05_positional_encoding.png` |
| 06 | [06_self_attention.md](06_self_attention.md) | `src/nanogpt/attention.py` | `assets/06_self_attention.png` |
| 07 | [07_multi_head_attention.md](07_multi_head_attention.md) | `src/nanogpt/multi_head_attention.py` | `assets/07_multi_head_attention.png` |
| 08 | [08_feed_forward.md](08_feed_forward.md) | `src/nanogpt/feed_forward.py` | `assets/08_feed_forward.png` |
| 09 | [09_layernorm_residual.md](09_layernorm_residual.md) | `src/nanogpt/layer_norm.py` | `assets/09_layernorm_residual.png` |
| 10 | [10_transformer_block.md](10_transformer_block.md) | `src/nanogpt/block.py` | `assets/10_transformer_block.png` |
| 11 | [11_gpt_model.md](11_gpt_model.md) | `src/nanogpt/model.py` | `assets/11_gpt_model.png` |
| 12 | [12_training.md](12_training.md) | `src/nanogpt/train.py` | `assets/12_training.png` |
| 13 | [13_sampling.md](13_sampling.md) | `src/nanogpt/generate.py` | `assets/13_sampling.png` |

## Interactive site

| Doc | What's inside |
|---|---|
| [14_web_visualizer.md](14_web_visualizer.md) | The `web/` interactive visualizer: structure, how widgets are wired, GitHub Pages hosting. All nine component widgets (03–11) are live. |

## Background

The starting-point diagrams are [`assets/00_rnn.png`](assets/00_rnn.png) (the RNN and its
limitations) and [`assets/01_transformer_architecture.png`](assets/01_transformer_architecture.png)
(the full Transformer), discussed in [01_architecture.md](01_architecture.md).

> **Note on the original diagram:** the classic *Attention Is All You Need* figure shows an
> **encoder–decoder** model. We build the **decoder-only** variant (GPT) — the right-hand
> stack, minus the encoder and the cross-attention. See [01_architecture.md](01_architecture.md#decoder-only).
