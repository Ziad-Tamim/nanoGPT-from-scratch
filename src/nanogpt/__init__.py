"""nanoGPT from scratch — a decoder-only Transformer, one component per file.

Each module implements a single Transformer component by hand (no torch.nn.Transformer)
and is paired with a doc page in docs/ and a diagram in docs/assets/. See docs/00_plan.md
for the build roadmap.
"""

from nanogpt.config import GPTConfig, get_device

__all__ = ["GPTConfig", "get_device"]
