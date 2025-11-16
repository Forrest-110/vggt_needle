#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from vggt.layers.rope import RotaryPositionEmbedding2D as NeedleRoPE

from needle import Tensor

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
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


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
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
    
def main():
    # -------------------------
    # Config
    # -------------------------
    torch.manual_seed(0)
    np.random.seed(0)

    B = 2          # batch size
    H = 3          # n_heads
    N = 16         # n_tokens
    D = 64         # feature dim, must be divisible by 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # -------------------------
    # Create shared numpy inputs
    # -------------------------
    # tokens: (B, H, N, D)
    tokens_np = np.random.randn(B, H, N, D).astype("float32")

    # positions: (B, N, 2)  y,x coordinates
    # here: simple regular grid for a HxW image, where H*W = N
    # for convenience choose H_img * W_img = N
    H_img = 4
    W_img = N // H_img
    assert H_img * W_img == N

    # build a (N, 2) grid, then broadcast to batch
    ys, xs = np.meshgrid(np.arange(H_img), np.arange(W_img), indexing="ij")
    grid = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)  # (N, 2)
    positions_np = np.broadcast_to(grid[None, :, :], (B, N, 2)).astype("int64")

    # -------------------------
    # Torch inputs
    # -------------------------
    tokens_torch = torch.tensor(tokens_np, device=device, dtype=dtype)
    positions_torch = torch.tensor(positions_np, device=device)

    # -------------------------
    # Needle inputs
    # -------------------------
    tokens_needle = Tensor(tokens_np)        # backend will decide device
    positions_needle = Tensor(positions_np)  # dtype can be int or float; we only use as indices/numbers

    # -------------------------
    # Instantiate RoPE modules
    # -------------------------
    base_frequency = 100.0
    scaling_factor = 1.0

    rope_torch = RotaryPositionEmbedding2D(frequency=base_frequency, scaling_factor=scaling_factor).to(device=device, dtype=dtype)
    rope_needle = NeedleRoPE(frequency=base_frequency, scaling_factor=scaling_factor)

    # -------------------------
    # Forward pass
    # -------------------------
    out_torch = rope_torch(tokens_torch, positions_torch)          # (B, H, N, D)
    out_needle = rope_needle(tokens_needle, positions_needle)      # (B, H, N, D) as Needle Tensor

    out_needle_np = out_needle.numpy()
    out_needle_torch = torch.from_numpy(out_needle_np).to(device=device, dtype=dtype)

    # -------------------------
    # Compare
    # -------------------------
    diff = (out_torch - out_needle_torch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("Torch output shape: ", tuple(out_torch.shape))
    print("Needle output shape:", tuple(out_needle_torch.shape))
    print("Max |diff|:  ", max_diff)
    print("Mean |diff|: ", mean_diff)

    # you can tighten or relax this tolerance if needed
    tol = 1e-5
    if max_diff < tol:
        print(f"[OK] RoPE match within tolerance {tol}")
    else:
        print(f"[WARN] RoPE mismatch: max diff {max_diff} > {tol}")


if __name__ == "__main__":
    main()