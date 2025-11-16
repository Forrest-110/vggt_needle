# test_mlp.py
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from needle import Tensor, nn
from vggt.layers.mlp import Mlp   # <- adjust if this class lives in a different module


def mlp_forward_numpy(x, W1, b1, W2, b2):
    """
    Reference forward for MLP with ReLU, no dropout:
        y = ReLU(x @ W1 + b1) @ W2 + b2
    Shapes:
        x: (B, D_in)
        W1: (D_in, D_hidden)
        b1: (1, D_hidden)
        W2: (D_hidden, D_out)
        b2: (1, D_out)
    """
    h = x @ W1.transpose(1,0) + b1          # (B, D_hidden)
    r = np.maximum(h, 0.0)   # ReLU
    y = r @ W2.transpose(1,0) + b2          # (B, D_out)
    return y, h, r


def test_mlp_forward():
    np.random.seed(0)

    B = 4
    D_in = 6
    D_hidden = 10
    D_out = 8

    # Needle MLP with ReLU, no dropout
    mlp = Mlp(
        in_features=D_in,
        hidden_features=D_hidden,
        out_features=D_out,
        act_layer=nn.ReLU,
        drop=0.0,
        bias=True,
    )

    x_np = np.random.randn(B, D_in).astype("float32")
    x = Tensor(x_np)

    # Needle forward
    y = mlp(x).numpy()

    # Grab parameters
    W1 = mlp.fc1.weight.numpy()       # (D_in, D_hidden)
    b1 = mlp.fc1.bias.numpy()         # (1, D_hidden)
    W2 = mlp.fc2.weight.numpy()       # (D_hidden, D_out)
    b2 = mlp.fc2.bias.numpy()         # (1, D_out)

    # NumPy reference
    y_ref, _, _ = mlp_forward_numpy(x_np, W1, b1, W2, b2)

    max_diff = np.max(np.abs(y - y_ref))
    print("Forward max diff:", max_diff)
    assert np.allclose(y, y_ref, atol=1e-6, rtol=1e-6), "Forward mismatch vs NumPy MLP"
    print("✓ Mlp forward matches NumPy reference.")


if __name__ == "__main__":
    test_mlp_forward()
    print("All Mlp tests passed ✅")