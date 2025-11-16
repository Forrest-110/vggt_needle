#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch
import torch.nn as nn_torch

from needle import Tensor, ops, init
from needle.nn import Conv


# ---- Test utilities ----

def make_aligned_convs(
    in_channels, out_channels, kernel_size, stride=1, bias=True
):
    """Create a PyTorch Conv2d and Needle Conv with IDENTICAL parameters."""
    torch_conv = nn_torch.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
    )

    needle_conv = Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        device=None,
        dtype="float32",
    )

    # Copy weights: torch (O, I, H, W) -> needle (H, W, I, O)
    with torch.no_grad():
        w_torch = torch_conv.weight.detach().cpu().numpy()  # (O, I, H, W)
        # w_hwio = np.transpose(w_torch, (2, 3, 1, 0))        # (H, W, I, O)

        needle_conv.weight.data = Tensor(w_torch.astype("float32"))

        if bias and torch_conv.bias is not None:
            b_torch = torch_conv.bias.detach().cpu().numpy()  # (O,)
 
            needle_conv.bias.data = Tensor(b_torch.astype("float32"))

    return torch_conv, needle_conv


def run_single_case(
    B, C_in, H, W, C_out, kernel_size, stride=1, bias=True,
    atol=1e-5, rtol=1e-5
):
    np.random.seed(0)
    torch.manual_seed(0)

    # Create aligned convs
    torch_conv, needle_conv = make_aligned_convs(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
    )

    # Random input
    x_np = np.random.randn(B, C_in, H, W).astype("float32")

    x_torch = torch.tensor(x_np, requires_grad=True)
    x_needle = Tensor(x_np, requires_grad=True)

    # Forward
    y_torch = torch_conv(x_torch)
    y_needle = needle_conv(x_needle)

    y_torch_np = y_torch.detach().cpu().numpy()
    y_needle_np = y_needle.numpy()

    # Compare forward
    max_diff = np.max(np.abs(y_torch_np - y_needle_np))
    print(f"[Forward] max diff: {max_diff:.6e}")
    assert np.allclose(y_torch_np, y_needle_np, atol=atol, rtol=rtol), "Forward mismatch!"

    print("✓ Case passed.\n")


if __name__ == "__main__":
    # A few different configurations to stress test
    cases = [
        # B, C_in, H, W, C_out, kernel_size, stride, bias
        (2, 3, 8, 8, 4, 3, 1, True),
        (2, 3, 9, 9, 5, 3, 2, True),
        (1, 1, 5, 5, 2, 1, 1, False),
        (4, 8, 16, 16, 8, 5, 1, True),
    ]

    for cfg in cases:
        print("Testing case:", cfg)
        run_single_case(*cfg)

    print("All Conv alignment tests passed ✅")