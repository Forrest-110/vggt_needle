# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
from needle import nn, Tensor, init, ops
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device=None) -> Tensor:
        # Cache key
        key = (height, width)

        if key not in self.position_cache:
            # y_coords: (H,)
            y_coords = init.arange(height, device=device)
            # x_coords: (W,)
            x_coords = init.arange(width, device=device)

            # Reshape for cartesian product:
            # y: (H, 1)
            # x: (1, W)
            y = y_coords.reshape((height, 1))
            x = x_coords.reshape((1, width))

            # Broadcast:
            # y_grid: (H, W)
            # x_grid: (H, W)
            y_grid = y.broadcast_to((height, width))
            x_grid = x.broadcast_to((height, width))

            # Stack → (H, W, 2)
            grid = ops.stack([y_grid, x_grid], axis=-1)

            # Flatten → (H*W, 2)
            grid = grid.reshape((height * width, 2))

            # Save in cache
            self.position_cache[key] = grid

        # Fetch cached (H*W, 2)
        positions = self.position_cache[key]

        # Expand to (batch_size, H*W, 2)
        # First reshape to (1, H*W, 2)
        positions = positions.reshape((1, height * width, 2))

        # Broadcast batch dim
        positions = positions.broadcast_to((batch_size, height * width, 2))

        return positions

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

class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.inv_freq_cache: Dict[Tuple[int, str], Tensor] = {}

    def _get_inv_freq(self, dim: int, device) -> Tensor:
        """
        dim: feature dimension for ONE spatial direction (vertical or horizontal).
             Must be even; number of frequency bands = dim // 2.
        Returns an NDArray of shape (dim // 2,)
        """
        key = (dim, str(device))
        if key not in self.inv_freq_cache:
            # exponents: 0, 2, 4, ..., dim-2  -> length dim//2
            exponents = init.arange(0, dim, 2, device=device) / dim

            base =  init.constant(*(exponents.shape), c=self.base_frequency, device=device)
            freq = base ** exponents

            inv_freq = freq ** -1
            inv_freq = inv_freq * self.scaling_factor
            self.inv_freq_cache[key] = inv_freq
        return self.inv_freq_cache[key]

    def _apply_1d_rope(
        self,
        tokens: Tensor,     # (B, H, N, D_dir)
        positions: Tensor,  # (B, N)
        inv_freq: Tensor
    ) -> Tensor:
        """
        Apply 1D RoPE to tokens using positions and inverse frequencies.

        We view the last dim D_dir as 2 * (D_dir/2) complex pairs:
          tokens -> (..., 2, D_dir/2)
        and apply a complex rotation with angles determined by positions * inv_freq.
        """
        B, H, N, D = tokens.shape
        assert D % 2 == 0, "Per-direction feature dim must be even for RoPE."
        half = D // 2  # number of complex pairs

        # positions: (B, N) -> NDArray
        pos_nd = positions
        pos_flat = pos_nd.reshape((B * N, 1))  # (B*N, 1)

        # inv_freq: (half,) -> (1, half)
        inv = inv_freq.reshape((1, half))

        # angles: (B*N, half) -> (B, N, half)
        angles = pos_flat.broadcast_to((pos_flat.shape[0], inv.shape[1])) * inv.broadcast_to((pos_flat.shape[0], inv.shape[1]))
        angles = angles.reshape((B, N, half))

        cos_nd = ops.cos(angles)
        sin_nd = ops.sin(angles)

        cos = cos_nd
        sin = sin_nd

        # Reshape cos/sin for broadcasting with tokens:
        # cos/sin: (B, 1, N, 1, half) -> broadcast to (B, H, N, 1, half)
        cos = ops.reshape(cos, (B, 1, N, 1, half))
        sin = ops.reshape(sin, (B, 1, N, 1, half))
        cos = ops.broadcast_to(cos, (B, H, N, 1, half))
        sin = ops.broadcast_to(sin, (B, H, N, 1, half))

        # tokens: (B, H, N, D) -> (B, H, N, half)
        x1, x2 = split_last_dim_in_half(tokens)

        # Add a singleton axis to match cos/sin shape: (B, H, N, 1, half)
        x1 = ops.reshape(x1, (B, H, N, 1, half))
        x2 = ops.reshape(x2, (B, H, N, 1, half))

        # Complex rotation:
        #   y1 = x1 * cos - x2 * sin
        #   y2 = x2 * cos + x1 * sin
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        # Stack back the two components along the "2" axis: (B, H, N, 2, half)
        stacked = ops.stack([y1, y2], axis=-2)

        # Flatten (2, half) -> D
        out = ops.reshape(stacked, (B, H, N, D))
        return out

    # ---------- forward: 2D RoPE (vertical + horizontal) ----------

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """
        tokens: (B, n_heads, n_tokens, dim), dim divisible by 4
        positions: (B, n_tokens, 2) with integer (y, x) coords
        """
        assert tokens.shape[-1] % 4 == 0, "Feature dim must be divisible by 4 for 2D RoPE."
        assert positions.ndim == 3 and positions.shape[-1] == 2, \
            "Positions must have shape (batch_size, n_tokens, 2)."

        B, H, N, D_total = tokens.shape
        feature_dim = D_total // 2  # per direction (vertical/horizontal)
        device = tokens.device

        # Get inverse frequencies for ONE direction (shared across batch/heads)
        inv_freq = self._get_inv_freq(feature_dim, device=device)

        # Split features into vertical / horizontal halves
        vertical_features, horizontal_features = split_last_dim_in_half(tokens)

        # positions_y/x: (B, N)
        positions_y, positions_x = split_last_dim_in_half(positions)

        # Apply 1D RoPE along each dimension
        vertical_rot = self._apply_1d_rope(vertical_features, positions_y, inv_freq)
        horizontal_rot = self._apply_1d_rope(horizontal_features, positions_x, inv_freq)

        # Combine back into (B, H, N, D_total) using stack + reshape
        vh = ops.stack([vertical_rot, horizontal_rot], axis=-2)   # (B, H, N, 2, feature_dim)
        out = ops.reshape(vh, (B, H, N, D_total))
        return out