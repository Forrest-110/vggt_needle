#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np

from needle import Tensor
from needle import nn
from vggt.layers.attention import Attention  # <-- change this if Attention is in another module


def attention_forward_numpy(attn: Attention, x_np: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of the Attention.forward logic,
    using the *actual* weights/biases from the needle Attention module.

    This assumes:
      - qk_norm=False
      - rope=None
    so that we don't have to reimplement LayerNorm or RoPE here.
    """
    B, N, C = x_np.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    scale = attn.scale
    # ---- qkv projection -------------------------------------------------
    # needle.nn.Linear is defined as: out = X @ W + b
    # where W has shape (in_features, out_features).
    W_qkv = attn.qkv.weight.numpy().transpose(1, 0)     # (C, 3C)
    b_qkv = attn.qkv.bias.numpy()       # (1, 3C) or (3C,)
    qkv = x_np @ W_qkv + b_qkv          # (B, N, 3C)

    # Reshape and split into q, k, v
    # qkv: (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).transpose(2, 0, 3, 1, 4)
    q_np, k_np, v_np = qkv[0], qkv[1], qkv[2]   # each: (B, num_heads, N, head_dim)

    # No qk_norm, no rope in this test
    # ---- scaled dot-product attention -----------------------------------
    q_np = q_np * scale                              # (B, H, N, D)
    # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
    attn_scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2))
    
    # softmax along the last dimension in a numerically stable way
    attn_scores = attn_scores - attn_scores.max(axis=-1, keepdims=True)
    attn_exp = np.exp(attn_scores)
    attn_np = attn_exp / attn_exp.sum(axis=-1, keepdims=True)   # (B, H, N, N)
    
    # ---- apply attention to values --------------------------------------
    # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
    x_np_attn = np.matmul(attn_np, v_np)

    # reshape back to (B, N, C)
    x_np_attn = x_np_attn.transpose(0, 2, 1, 3).reshape(B, N, C)
    
    # ---- output projection ----------------------------------------------
    W_proj = attn.proj.weight.numpy().transpose(1, 0)   # (C, C)
    b_proj = attn.proj.bias.numpy()     # (1, C) or (C,)

    out_np = x_np_attn @ W_proj + b_proj  # (B, N, C)

    return out_np


def test_attention_forward_shape():
    np.random.seed(0)
    B, N, C = 2, 4, 16
    x_np = np.random.randn(B, N, C).astype("float32")
    x = Tensor(x_np)
    attn = Attention(
        dim=C,
        num_heads=4,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=False,
        rope=None,
    )

    y = attn(x)  # Tensor
    y_np = y.numpy()

    assert y_np.shape == (B, N, C), f"Expected shape {(B, N, C)}, got {y_np.shape}"
    print("[OK] test_attention_forward_shape")


def test_attention_forward_numeric():
    """
    Compare needle Attention forward with an explicit NumPy implementation,
    using the SAME weights & biases taken from the needle module.

    This only tests the case qk_norm=False and rope=None.
    """
    np.random.seed(1)
    B, N, C = 3, 5, 32
    x_np = np.random.randn(B, N, C).astype("float32")
    x = Tensor(x_np)

    attn = Attention(
        dim=C,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=False,
        rope=None,
    )

    # needle output
    y = attn(x).numpy()

    # NumPy reference output
    y_ref = attention_forward_numpy(attn, x_np)

    max_diff = np.max(np.abs(y - y_ref))
    print("Max abs diff (needle vs NumPy):", max_diff)
    assert np.allclose(y, y_ref, atol=1e-5, rtol=1e-5), \
        f"Forward mismatch: max diff {max_diff}"
    print("[OK] test_attention_forward_numeric")



if __name__ == "__main__":
    # test_attention_forward_shape()
    test_attention_forward_numeric()
    print("All Attention tests passed ✅")