# nanoGPT From Scratch — Learning the Transformer

A from-scratch, **decoder-only Transformer (GPT)** built one component per file, with
rigorous math notes and hand-drawn (Excalidraw) diagrams for every piece. The goal is
**understanding**, not just a working model: each module is small, documented, and paired
with a `docs/` page explaining the math and a diagram in `docs/assets/`.

Trains a character-level language model on **Tiny Shakespeare** and generates text.

| Choice | Decision |
|---|---|
| Framework | PyTorch (tensors + autograd), but every component (attention, positional encoding, layer norm, ...) is written **by hand** — no `nn.Transformer`. |
| Dataset | Tiny Shakespeare (~1 MB), character-level |
| Tokenizer | Character-level (vocab ≈ 65) |
| Hardware | NVIDIA GPU (CUDA), with CPU fallback |
| Math depth | Rigorous — formulas, tensor shapes, gradient/backprop notes, paper references |

## Quick map

- 📋 **[docs/00_plan.md](docs/00_plan.md)** — the full build roadmap and step-by-step order
- 🏗️ **[docs/01_architecture.md](docs/01_architecture.md)** — file structure + how the pieces connect
- 📚 **[docs/README.md](docs/README.md)** — index of every component doc
- 🎨 **[docs/assets/](docs/assets/)** — exported Excalidraw diagrams (one per component)

## Status

🚧 Planning / scaffolding. See [docs/00_plan.md](docs/00_plan.md) for the roadmap and progress checklist.

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Karpathy, *nanoGPT* and *Let's build GPT*
