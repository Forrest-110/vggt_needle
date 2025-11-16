#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

# 🔧 Adjust these imports to your actual paths
# e.g. from needle.nn import MultiheadAttention
from needle import Tensor
from needle.nn import MultiheadAttention


def copy_torch_mha_to_needle(torch_mha: torch.nn.MultiheadAttention,
                             needle_mha: MultiheadAttention):
    """
    Copy parameters from a torch.nn.MultiheadAttention to a Needle MultiheadAttention.

    Assumptions:
      - Needle MHA has q_proj, k_proj, v_proj, out_proj as nn.Linear-like modules
      - Needle Linear uses weight shape (in_features, out_features) and forward: X @ W
      - PyTorch uses in_proj_weight (3E, E) & in_proj_bias (3E,)
        and out_proj.weight (E, E) with F.linear(input, weight, bias) = input @ weight.T
    """
    E = torch_mha.embed_dim

    # --- in_proj (Q, K, V) ---
    with torch.no_grad():
        in_w = torch_mha.in_proj_weight.detach().cpu().numpy()  # (3E, E)
        in_b = torch_mha.in_proj_bias.detach().cpu().numpy()    # (3E,)

        # q_w_torch = in_w[0:E, :]          # (E, E)
        # k_w_torch = in_w[E:2*E, :]
        # v_w_torch = in_w[2*E:3*E, :]
        # q_b_torch = in_b[0:E]
        # k_b_torch = in_b[E:2*E]
        # v_b_torch = in_b[2*E:3*E]

        # Needle Linear: weight (in_features, out_features), forward: X @ W
        # PyTorch Linear: weight (out_features, in_features), forward: X @ W^T
        # => needle_weight = torch_weight.T
        needle_mha.in_proj_weight = Tensor(in_w.astype("float32"))
        # needle_mha.k_proj.weight = Tensor(k_w_torch.astype("float32"))
        # needle_mha.v_proj.weight = Tensor(v_w_torch.astype("float32"))

        needle_mha.in_proj_bias = Tensor(in_b.astype("float32"))
        # needle_mha.k_proj.bias = Tensor(k_b_torch.astype("float32"))
        # needle_mha.v_proj.bias = Tensor(v_b_torch.astype("float32"))

        # --- out_proj ---
        out_w = torch_mha.out_proj.weight.detach().cpu().numpy()  # (E, E)
        out_b = torch_mha.out_proj.bias.detach().cpu().numpy()    # (E,)

        needle_mha.out_proj.weight = Tensor(out_w.astype("float32"))
        needle_mha.out_proj.bias = Tensor(out_b.astype("float32"))


def needle_to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy()


def run_single_test(embed_dim=32, num_heads=4, B=2, T_q=5, T_k=7):
    torch.manual_seed(0)
    np.random.seed(0)

    # ----- Build PyTorch MHA -----
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,          # (B, T, E)
        bias=True,
    )

    # ----- Build Needle MHA -----
    needle_mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        dtype="float32",
    )

    # Copy parameters from torch -> needle
    copy_torch_mha_to_needle(torch_mha, needle_mha)

    # ----- Create inputs -----
    # Self-attention: q = k = v
    q_t = torch.randn(B, T_q, embed_dim, dtype=torch.float32)
    k_t = torch.randn(B, T_k, embed_dim, dtype=torch.float32)
    v_t = torch.randn(B, T_k, embed_dim, dtype=torch.float32)

    q_n = Tensor(q_t.detach().cpu().numpy())
    k_n = Tensor(k_t.detach().cpu().numpy())
    v_n = Tensor(v_t.detach().cpu().numpy())

    # ----- Forward: PyTorch -----
    # average_attn_weights=False to get per-head weights
    out_t, attn_t = torch_mha(
        q_t, k_t, v_t,
        need_weights=True,
        average_attn_weights=False,
    )
    # out_t: (B, T_q, E)
    # attn_t: (B * num_heads, T_q, T_k)
    attn_t = attn_t.detach().cpu()
    out_t = out_t.detach().cpu()

    # reshape attn_t -> (B, H, T_q, T_k)
    attn_t = attn_t.view(B, num_heads, T_q, T_k)

    # ----- Forward: Needle -----
    out_n, attn_n = needle_mha(q_n, k_n, v_n, need_weights=True)
    out_n_np = needle_to_numpy(out_n)
    attn_n_np = needle_to_numpy(attn_n)

    # ----- Compare shapes -----
    assert out_n_np.shape == tuple(out_t.shape), \
        f"Output shape mismatch: needle {out_n_np.shape}, torch {tuple(out_t.shape)}"
    assert attn_n_np.shape == tuple(attn_t.shape), \
        f"Attn shape mismatch: needle {attn_n_np.shape}, torch {tuple(attn_t.shape)}"

    # ----- Compare values -----
    np.testing.assert_allclose(
        out_n_np, out_t.numpy(), rtol=1e-5, atol=1e-6,
        err_msg="MultiheadAttention output mismatch",
    )
    np.testing.assert_allclose(
        attn_n_np, attn_t.numpy(), rtol=1e-5, atol=1e-6,
        err_msg="MultiheadAttention attention weights mismatch",
    )

    print(f"✅ Passed: embed_dim={embed_dim}, num_heads={num_heads}, "
          f"B={B}, T_q={T_q}, T_k={T_k}")


def test_self_attention():
    """q = k = v case."""
    print("Testing self-attention (q = k = v)…")
    embed_dim = 32
    num_heads = 4
    B, T = 2, 5

    torch.manual_seed(1)
    np.random.seed(1)

    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
        bias=True,
    )
    needle_mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        dtype="float32",
    )
    copy_torch_mha_to_needle(torch_mha, needle_mha)

    x_t = torch.randn(B, T, embed_dim, dtype=torch.float32)
    x_n = Tensor(x_t.detach().cpu().numpy())

    out_t, attn_t = torch_mha(
        x_t, x_t, x_t,
        need_weights=True,
        average_attn_weights=False,
    )
    out_t = out_t.detach().cpu()
    attn_t = attn_t.detach().cpu()
    attn_t = attn_t.view(B, num_heads, T, T)

    out_n, attn_n = needle_mha(x_n, x_n, x_n, need_weights=True)
    out_n_np = needle_to_numpy(out_n)
    attn_n_np = needle_to_numpy(attn_n)

    assert out_n_np.shape == tuple(out_t.shape)
    assert attn_n_np.shape == tuple(attn_t.shape)

    np.testing.assert_allclose(
        out_n_np, out_t.numpy(), rtol=1e-5, atol=1e-6,
        err_msg="Self-attention output mismatch",
    )
    np.testing.assert_allclose(
        attn_n_np, attn_t.numpy(), rtol=1e-5, atol=1e-6,
        err_msg="Self-attention attn weights mismatch",
    )

    print("✅ Self-attention test passed.")


if __name__ == "__main__":
    # Cross-attention style test (T_q != T_k)
    run_single_test(embed_dim=32, num_heads=4, B=2, T_q=5, T_k=7)

    # Self-attention test (q = k = v)
    test_self_attention()

    print("\nAll MultiheadAttention alignment tests passed ✅")
