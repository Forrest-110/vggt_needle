#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np

from needle import Tensor

# 🔧 Adjust this to your actual file path
# e.g. from vggt.blocks.attn_blocks import Mlp, AttnBlock, CrossAttnBlock
from vggt.heads.track_modules.modules import Mlp, AttnBlock, CrossAttnBlock


def test_mlp_linear():
    """
    Test Mlp in pure linear mode (use_conv=False).
    Input: (B, T, C)
    """
    print("Testing Mlp (linear mode)...")
    B, T, C = 2, 5, 16
    hidden = 32
    x_np = np.random.randn(B, T, C).astype("float32")
    x = Tensor(x_np)

    mlp = Mlp(
        in_features=C,
        hidden_features=hidden,
        out_features=C,
        use_conv=False,
        drop=0.0,
    )

    y = mlp(x)
    y_np = y.numpy()
    assert np.isfinite(y_np).all()
    assert y_np.shape == x_np.shape, f"Mlp (linear) shape mismatch: {y_np.shape} vs {x_np.shape}"
    print("  ✅ Mlp (linear) forward OK; output shape:", y_np.shape)


def test_mlp_conv():
    """
    Test Mlp in conv mode (use_conv=True).
    Here Mlp uses 1x1 convolutions, so we feed NCHW.
    """
    print("Testing Mlp (conv mode)...")
    N, C, H, W = 2, 16, 8, 8
    hidden = 32
    x_np = np.random.randn(N, C, H, W).astype("float32")
    x = Tensor(x_np)

    mlp = Mlp(
        in_features=C,
        hidden_features=hidden,
        out_features=C,
        use_conv=True,
        drop=0.0,
    )

    y = mlp(x)
    y_np = y.numpy()
    assert np.isfinite(y_np).all()
    assert y_np.shape == x_np.shape, f"Mlp (conv) shape mismatch: {y_np.shape} vs {x_np.shape}"
    print("  ✅ Mlp (conv) forward OK; output shape:", y_np.shape)


def test_attn_block():
    """
    Test AttnBlock forward.
    Input: (B, T, hidden_size)
    """
    print("Testing AttnBlock...")
    B, T, hidden_size, num_heads = 2, 6, 32, 4
    x_np = np.random.randn(B, T, hidden_size).astype("float32")
    x = Tensor(x_np)

    block = AttnBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )

    y = block(x)  # mask=None
    y_np = y.numpy()
    assert np.isfinite(y_np).all()
    assert y_np.shape == x_np.shape, f"AttnBlock output shape mismatch: {y_np.shape} vs {x_np.shape}"
    print("  ✅ AttnBlock forward OK; output shape:", y_np.shape)


def test_cross_attn_block():
    """
    Test CrossAttnBlock forward.
    - x: (B, T_q, hidden_size)
    - context: (B, T_k, hidden_size)
    """
    print("Testing CrossAttnBlock...")
    B = 2
    T_q = 5
    T_k = 7
    hidden_size = 32
    num_heads = 4

    x_np = np.random.randn(B, T_q, hidden_size).astype("float32")
    ctx_np = np.random.randn(B, T_k, hidden_size).astype("float32")

    x = Tensor(x_np)
    context = Tensor(ctx_np)

    block = CrossAttnBlock(
        hidden_size=hidden_size,
        context_dim=hidden_size,
        num_heads=num_heads,
    )

    # mask=None; CrossAttnBlock will call cross_attn(x, context, context, attn_mask=None)
    y = block(x, context, mask=None)
    y_np = y.numpy()
    assert np.isfinite(y_np).all()
    assert y_np.shape == x_np.shape, f"CrossAttnBlock output shape mismatch: {y_np.shape} vs {x_np.shape}"
    print("  ✅ CrossAttnBlock forward OK; output shape:", y_np.shape)


def main():
    np.random.seed(0)
    test_mlp_linear()
    test_mlp_conv()
    test_attn_block()
    test_cross_attn_block()
    print("\nAll block forward tests passed ✅")


if __name__ == "__main__":
    main()
