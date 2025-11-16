#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from needle import nn as nn_custom
from torch import nn as nn_torch

import torch
import needle
import numpy as np


def compare_layernorms(shape, dim, eps=1e-5, atol=1e-6, rtol=1e-6):
    print(f"\n=== Testing shape={shape}, dim={dim} ===")
    np.random.seed(0)
    x_np = np.random.randn(*shape)
    # random input
    x_ref = torch.from_numpy(x_np).float()
    x_my  = needle.Tensor(x_np)
    # reference LayerNorm
    ln_ref = nn_torch.LayerNorm(normalized_shape=dim, eps=eps, elementwise_affine=True)

    # our custom LayerNorm with same hyperparams
    ln_my = nn_custom.LayerNorm(dim=dim, eps=eps)

    # copy parameters so they start identical
    with torch.no_grad():
        ln_my.weight.data = needle.Tensor(ln_ref.weight.cpu().numpy()) 
        ln_my.bias.data = needle.Tensor(ln_ref.bias.cpu().numpy())

    # forward
    y_ref = ln_ref(x_ref)
    y_my  = ln_my(x_my)

    # compare outputs
    max_abs_diff = np.abs(y_ref.detach().numpy() - y_my.numpy()).max()
    print(f"max |y_ref - y_my| = {max_abs_diff:.3e}")


def compare_linears(shape, out_features, bias=True, atol=1e-6, rtol=1e-6):
    print(f"\n=== Testing Linear: shape={shape}, out_features={out_features}, bias={bias} ===")

    np.random.seed(0)
    x_np = np.random.randn(*shape).astype("float32")

    # infer in_features from last dim
    in_features = shape[-1]

    # random input
    x_ref = torch.from_numpy(x_np).float()
    x_my  = needle.Tensor(x_np)

    # reference Linear (PyTorch)
    lin_ref = nn_torch.Linear(in_features, out_features, bias=bias)

    # our custom Linear with same hyperparams
    lin_my = nn_custom.Linear(in_features, out_features, bias=bias)

    # copy parameters so they start identical
    with torch.no_grad():
        # weight: (out_features, in_features)
        w_ref = lin_ref.weight.detach().cpu().numpy()
        lin_my.weight.data = needle.Tensor(w_ref)

        if bias:
            b_ref = lin_ref.bias.detach().cpu().numpy()
            lin_my.bias.data = needle.Tensor(b_ref)

    # forward
    y_ref = lin_ref(x_ref)      # torch.Tensor
    y_my  = lin_my(x_my)        # needle.Tensor

    # compare outputs
    y_ref_np = y_ref.detach().cpu().numpy()
    y_my_np  = y_my.numpy()

    max_abs_diff = np.max(np.abs(y_ref_np - y_my_np))
    rel_diff = np.max(np.abs(y_ref_np - y_my_np) / (np.abs(y_ref_np) + 1e-12))

    print(f"max |y_ref - y_my| = {max_abs_diff:.3e}")
    print(f"max relative diff  = {rel_diff:.3e}")

    assert np.allclose(y_ref_np, y_my_np, atol=atol, rtol=rtol), \
        f"Linear forward mismatch: max_abs_diff={max_abs_diff}, rel_diff={rel_diff}"

if __name__ == "__main__":
    dim = 16
    shapes = [
        (4, dim),        # 2D: (B, D)
        (2, 3, dim),     # 3D: (B, T, D)
        (1, 5, 7, dim),  # 4D: arbitrary leading dims
    ]
    for shape in shapes:
        compare_layernorms(shape, dim=dim)
        compare_linears(shape, 128, True)
        compare_linears(shape, 256, False)
    print("\nAll tests passed.")