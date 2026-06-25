"""Smoke-check the environment: PyTorch, CUDA, and GPTConfig.

Run after `uv sync`:

    uv run python scripts/check_setup.py
"""

import torch

from nanogpt import GPTConfig, get_device


def main() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"Selected device : {get_device()}")

    cfg = GPTConfig()
    print(
        f"GPTConfig       : n_embd={cfg.n_embd}, n_head={cfg.n_head}, "
        f"head_size={cfg.head_size}, n_layer={cfg.n_layer}, block_size={cfg.block_size}"
    )

    # A tiny tensor round-trip on the chosen device proves the install works end to end.
    device = cfg.resolve_device()
    x = torch.randn(2, cfg.block_size, cfg.n_embd, device=device)
    print(f"Test tensor     : {tuple(x.shape)} on {x.device}  ->  mean {x.mean().item():.4f}")
    print("\nSetup looks good.")


if __name__ == "__main__":
    main()
