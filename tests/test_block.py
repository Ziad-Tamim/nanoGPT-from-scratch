"""Tests for the Transformer block. Run: uv run python -m pytest tests/ -q"""

import torch

from nanogpt.block import Block


def make_block(n_embd=32, n_head=4, block_size=8, dropout=0.0):
    return Block(n_embd, n_head, block_size, dropout).eval()


def test_output_shape_preserved():
    blk = make_block()
    x = torch.randn(4, 8, 32)
    assert blk(x).shape == (4, 8, 32)


def test_returns_attention_weights():
    blk = make_block(n_head=4, block_size=8)
    out, w = blk(torch.randn(2, 8, 32), return_weights=True)
    assert out.shape == (2, 8, 32)
    assert w.shape == (2, 4, 8, 8)  # (B, n_head, T, T)


def test_causality_survives_full_block():
    blk = make_block(block_size=8)
    x = torch.randn(1, 8, 32)
    out1 = blk(x)
    x2 = x.clone()
    x2[0, 6] = torch.randn(32)
    out2 = blk(x2)
    # Outputs before the perturbed position must be unchanged.
    assert torch.allclose(out1[0, :6], out2[0, :6], atol=1e-6)


def test_gradients_reach_all_subcomponents():
    blk = make_block()
    blk(torch.randn(2, 8, 32)).sum().backward()
    assert blk.ln1.gamma.grad.abs().sum() > 0
    assert blk.attn.proj.weight.grad.abs().sum() > 0
    assert blk.ln2.gamma.grad.abs().sum() > 0
    assert blk.ffwd.net[0].weight.grad.abs().sum() > 0


def test_residual_path_present():
    """With zeroed sublayer output projections, the block should approximate identity."""
    blk = make_block()
    with torch.no_grad():
        blk.attn.proj.weight.zero_(); blk.attn.proj.bias.zero_()
        blk.ffwd.net[2].weight.zero_(); blk.ffwd.net[2].bias.zero_()
    x = torch.randn(1, 8, 32)
    assert torch.allclose(blk(x), x, atol=1e-6)  # only the residual remains
