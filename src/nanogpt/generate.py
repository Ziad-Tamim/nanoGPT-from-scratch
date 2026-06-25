"""Text generation — load a trained checkpoint and sample new text.

See docs/13_sampling.md.

Generation is autoregressive: predict the next-token distribution, sample one token, append
it, and repeat. Run:

    uv run python -m nanogpt.generate --prompt "ROMEO:" --tokens 500 --temperature 0.8

Sampling knobs:
    --temperature  <1 sharpens (greedier), >1 flattens (more random)
    --top-k        restrict sampling to the k most likely tokens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanogpt.config import GPTConfig
from nanogpt.model import GPT
from nanogpt.tokenizer import CharTokenizer


def load_checkpoint(path: str | Path, device: str) -> tuple[GPT, CharTokenizer]:
    """Rebuild the model and tokenizer from a saved checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # disable dropout for deterministic-ish sampling
    tokenizer = CharTokenizer(ckpt["vocab"])
    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: CharTokenizer,
    device: str,
    prompt: str = "\n",
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int | None = 200,
) -> str:
    """Encode ``prompt``, sample ``max_new_tokens`` more tokens, and decode to text."""
    ids = tokenizer.encode(prompt) or [tokenizer.stoi.get("\n", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, len(prompt))
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained nanoGPT checkpoint.")
    p.add_argument("--ckpt", type=str, default="checkpoints/ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500, help="number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = args.device or GPTConfig().resolve_device()

    model, tokenizer = load_checkpoint(args.ckpt, device)
    text = generate_text(
        model, tokenizer, device,
        prompt=args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
