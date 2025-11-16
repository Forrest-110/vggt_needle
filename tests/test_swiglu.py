# test_mlp.py
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from needle import Tensor, nn
from vggt.layers.swiglu_ffn import SwiGLU


def test_mlp_forward():
    np.random.seed(0)

    B = 4
    D_in = 6
    D_hidden = 10
    D_out = 8

    # Needle MLP with ReLU, no dropout
    mlp = SwiGLU(
        in_features=D_in,
        hidden_features=D_hidden,
        out_features=D_out,
        bias=True,
    )

    x_np = np.random.randn(B, D_in).astype("float32")
    x = Tensor(x_np)

    # Needle forward
    y = mlp(x).numpy()

  

if __name__ == "__main__":
    test_mlp_forward()
    print("All Mlp tests passed ✅")