#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from needle import Tensor
from needle import nn


def test_gelu_forward():
    np.random.seed(0)
    torch.manual_seed(0)

    # random input
    x_np = np.random.randn(4, 7).astype("float32")

    # needle tensor
    x_needle = Tensor(x_np, requires_grad=False)

    # torch tensor
    x_torch = torch.tensor(x_np, requires_grad=False)

    gelu_needle = nn.GELU()
    gelu_torch = torch.nn.GELU()

    y_needle = gelu_needle(x_needle).numpy()
    y_torch = gelu_torch(x_torch).detach().cpu().numpy()

    max_diff = np.max(np.abs(y_needle - y_torch))
    print("Forward max diff:", max_diff)
    assert np.allclose(y_needle, y_torch, rtol=1e-5, atol=1e-6), "Forward mismatch vs nn.GELU(tanh)"
    print("✓ GELU forward matches torch.nn.GELU(approximate='tanh').")



if __name__ == "__main__":
    test_gelu_forward()
    print("All GELU tests passed ✅")