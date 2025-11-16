#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np

from needle import Tensor
from needle import nn
from vggt.layers.layer_scale import LayerScale



if __name__ == "__main__":
    import numpy as np

    # ---------------------------
    # Config
    # ---------------------------
    B, N, C = 2, 4, 8      # batch, tokens, channels
    init_value = 1e-3

    # ---------------------------
    # Build module
    # ---------------------------
    ls = LayerScale(dim=C, init_values=init_value)

    # ---------------------------
    # Make test input
    # ---------------------------
    x_np = np.random.randn(B, N, C).astype("float32")
    x = Tensor(x_np, requires_grad=True)

    # ---------------------------
    # Forward
    # ---------------------------
    y = ls(x)
    y_np = y.numpy()

    # ---------------------------
    # Reference using NumPy
    # ---------------------------
    gamma_np = ls.gamma.numpy().reshape((1, 1, C))  # broadcast shape
    y_ref = x_np * gamma_np

    # ---------------------------
    # Compare forward
    # ---------------------------
    max_diff = np.max(np.abs(y_np - y_ref))
    print("Forward max diff:", max_diff)
    assert np.allclose(y_np, y_ref, atol=1e-6), "Forward mismatch!"
    print("✓ Forward matches reference.")