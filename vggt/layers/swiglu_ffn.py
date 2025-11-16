# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Optional
import warnings

from needle import Tensor, nn, ops

def split_last_dim_in_half(x: Tensor):
    D = x.shape[-1]
    assert D % 2 == 0, "Last dim must be even for split_last_dim_in_half."

    half = D // 2

    # (B, H, N, D) -> (B, H, N, 2, half)
    x5 = ops.reshape(x, (*x.shape[:-1], 2, half))

    # unbind along the "2" axis -> two tensors of shape (B, H, N, half)
    x0, x1 = ops.unbind(x5, axis=-2)

    # already (B, H, N, half) so we can just return
    return x0, x1

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = split_last_dim_in_half(x12)
        hidden = ops.silu(x1) * x2
        return self.w3(hidden)



SwiGLU = SwiGLUFFN


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features, out_features=out_features, bias=bias)