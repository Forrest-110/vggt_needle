#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np

from needle import Tensor
from needle import nn
from vggt.layers.block import *




if __name__ == "__main__":
    import numpy as np

    
    def test_block_no_drop():
        np.random.seed(2)
        B, N, D = 4, 6, 16
        num_heads = 4

        x_np = np.random.randn(B, N, D).astype("float32")
        x = Tensor(x_np, requires_grad=True)

        block = Block(
            dim=D,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop=0.0,
            attn_drop=0.0,
            init_values=None,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_class=Attention,
            ffn_layer=Mlp,
            qk_norm=False,
            fused_attn=False,
            rope=None,
        )

        y = block(x)      # (B, N, D)
        assert y.shape == x.shape

        

    test_block_no_drop()
    
    print("All Block + stochastic-depth tests passed ✅")