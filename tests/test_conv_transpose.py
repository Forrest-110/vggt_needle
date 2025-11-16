#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from needle import Tensor
from needle import ops

from needle.nn import ConvTranspose2d  


# ------ Helpers ------

def needle_to_numpy(x: Tensor) -> np.ndarray:
    # Adjust this if your Tensor API is different
    return x.numpy()


def assert_allclose(a, b, rtol=1e-5, atol=1e-6, msg=""):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def copy_torch_weights_to_needle(conv_pt: torch.nn.ConvTranspose2d,
                                 conv_ndl: ConvTranspose2d):
    """
    Map PyTorch ConvTranspose2d weights/bias to Needle ConvTranspose2d.

    PyTorch: weight (C_in, C_out, K, K)
    Needle:  weight (K, K, C_out, C_in)
    """
    w_pt = conv_pt.weight.detach().cpu().numpy()     # (C_in, C_out, K, K)
    b_pt = conv_pt.bias.detach().cpu().numpy() if conv_pt.bias is not None else None

    # permute to Needle layout: (K, K, C_out, C_in)
    # w_ndl = np.transpose(w_pt, (2, 3, 1, 0))         # (K, K, C_out, C_in)

    conv_ndl.weight.data = Tensor(w_pt.copy())
    if b_pt is not None and conv_ndl.bias is not None:
        conv_ndl.bias.data = Tensor(b_pt.copy())


def run_single_case(
    N, C_in, C_out, H, W,
    kernel_size, stride, padding, output_padding=0,
    device="cpu"
):
    print(f"  Case: N={N}, C_in={C_in}, C_out={C_out}, "
          f"H={H}, W={W}, K={kernel_size}, s={stride}, p={padding}, op={output_padding}")

    torch.manual_seed(0)
    np.random.seed(0)

    # ----- PyTorch module -----
    conv_pt = torch.nn.ConvTranspose2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=device,
        dtype=torch.float32,
    )

    # ----- Needle module -----
    conv_ndl = ConvTranspose2d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=device,
        dtype="float32",
    )

    # copy weights & bias so both layers are identical
    copy_torch_weights_to_needle(conv_pt, conv_ndl)

    # ----- Input -----
    x_pt = torch.randn(N, C_in, H, W, device=device, dtype=torch.float32, requires_grad=True)
    x_np = x_pt.detach().cpu().numpy()
    x_ndl = Tensor(x_np, dtype="float32")
    x_ndl.requires_grad = True

    # ----- Forward -----
    y_pt = conv_pt(x_pt)             # (N, C_out, H_out, W_out)
    y_ndl = conv_ndl(x_ndl)          # (N, C_out, H_out, W_out)

    y_pt_np = y_pt.detach().cpu().numpy()
    y_ndl_np = needle_to_numpy(y_ndl)

    print("    forward: comparing outputs...")
    assert y_pt_np.shape == y_ndl_np.shape, f"{y_pt_np.shape}, {y_ndl_np.shape}"
    assert_allclose(y_ndl_np, y_pt_np, msg="Forward outputs mismatch")


    print("    ✓ case passed")


def test_conv_transpose2d():
    print("Testing Needle nn.ConvTranspose2d against torch.nn.ConvTranspose2d\n")

    cases = [
        # (N, C_in, C_out, H, W, K, stride, padding, output_padding)
        (2, 3, 4, 5, 5, 3, 1, 0, 0),
        (2, 3, 2, 7, 7, 3, 2, 1, 0),
        (1, 1, 1, 8, 8, 4, 2, 1, 0),
        (2, 4, 3, 6, 6, 5, 1, 2, 0),
    ]

    for cfg in cases:
        run_single_case(*cfg)

    print("\nAll ConvTranspose2d tests passed ✅")


if __name__ == "__main__":
    test_conv_transpose2d()
