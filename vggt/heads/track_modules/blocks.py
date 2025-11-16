# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Modified from https://github.com/facebookresearch/co-tracker/

import math
from needle import nn, ops, init, Tensor

from vggt.heads.track_modules.utils import bilinear_sampler
from vggt.heads.track_modules.modules import AttnBlock, CrossAttnBlock
from vggt.heads.utils import needle_meshgrid as meshgrid, needle_linspace as linspace


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
    ):
        super().__init__()

        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn

        # Add input LayerNorm before linear projection
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)

        # Add output LayerNorm before final projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        if self.add_space_attn:
            self.virual_tracks = nn.Parameter(init.randn(1, num_virtual_tracks, 1, hidden_size))
        else:
            self.virual_tracks = None

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
 

    def forward(self, input_tensor, mask=None):
        # Apply input LayerNorm
        input_tensor = self.input_norm(input_tensor)
        tokens = self.input_transform(input_tensor)

        init_tokens = tokens

        B, _, T, _ = tokens.shape

        if self.add_space_attn:
            virtual_tokens = self.virual_tracks.broadcast_to((B, self.virual_tracks.shape[1], T, self.virual_tracks.shape[3]))
            tokens = ops.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            tokens = tokens + 0.0
            time_tokens = tokens.reshape((B * N, T, -1))  # B N T C -> (B N) T C

            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.reshape((B, N, T, -1))  # (B N) T C -> B N T C
    
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0):
                space_tokens = (tokens.permute((0, 2, 1, 3))+0.0).reshape((B * T, N, -1))  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks, :]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :, :]

                virtual_tokens = self.space_virtual2point_blocks[j](virtual_tokens, point_tokens, mask=mask)
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](point_tokens, virtual_tokens, mask=mask)

                space_tokens = ops.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.reshape((B, T, N, -1)).permute((0, 2, 1, 3))  # (B T) N C -> B N T C
                j += 1
           

        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks, :, :]

        tokens = tokens + init_tokens

        # Apply output LayerNorm before final projection
        tokens = self.output_norm(tokens)
        flow = self.flow_head(tokens)

        return flow, None

import torch
class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4, multiple_track_feats=False, padding_mode="zeros"):
        """
        Build a pyramid of feature maps from the input.

        fmaps: Tensor (B, S, C, H, W)
        num_levels: number of pyramid levels (each downsampled by factor 2)
        radius: search radius for sampling correlation
        multiple_track_feats: if True, split the target features per pyramid level
        padding_mode: passed to grid_sample / bilinear_sampler
        """
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.num_levels = num_levels
        self.radius = radius
        self.padding_mode = padding_mode
        self.multiple_track_feats = multiple_track_feats

        # Build pyramid: each level is half the spatial resolution of the previous
        self.fmaps_pyramid = [fmaps]  # level 0 is full resolution
        current_fmaps = torch.from_numpy(fmaps.numpy())
        for i in range(num_levels - 1):
            B, S, C, H, W = current_fmaps.shape
            # Merge batch & sequence dimensions
            current_fmaps = current_fmaps.reshape(B * S, C, H, W)
            # Avg pool down by factor 2
            current_fmaps = torch.nn.functional.avg_pool2d(current_fmaps, kernel_size=2, stride=2)
            _, _, H_new, W_new = current_fmaps.shape
            current_fmaps = current_fmaps.reshape(B, S, C, H_new, W_new)
            self.fmaps_pyramid.append(Tensor(current_fmaps.numpy()))

        # Precompute a delta grid (of shape (2r+1, 2r+1, 2)) for sampling.
        # This grid is added to the (scaled) coordinate centroids.
        r = self.radius
        dx = linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        dy = linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        # delta: for every (dy,dx) displacement (i.e. Δx, Δy)
        self.delta = ops.stack(meshgrid(dy, dx, indexing="ij"), axis=-1)  # shape: (2r+1, 2r+1, 2)

    def corr_sample(self, targets, coords):
        """
        Instead of storing the entire correlation pyramid, we compute each level's correlation
        volume, sample it immediately, then discard it. This saves GPU memory.

        Args:
          targets: Tensor (B, S, N, C) — features for the current targets.
          coords: Tensor (B, S, N, 2) — coordinates at full resolution.

        Returns:
          Tensor (B, S, N, L) where L = num_levels * (2*radius+1)**2 (concatenated sampled correlations)
        """
        B, S, N, C = targets.shape

        # If you have multiple track features, split them per level.
        if self.multiple_track_feats:
            targets_split = ops.split(targets, C // self.num_levels, dim=-1)

        out_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            # Get current spatial resolution H, W for this pyramid level.
            B, S, C, H, W = fmaps.shape
            # Reshape feature maps for correlation computation:
            # fmap2s: (B, S, C, H*W)
            fmap2s = fmaps.reshape((B, S, C, H * W))
            # Choose appropriate target features.
            fmap1 = targets_split[i] if self.multiple_track_feats else targets  # shape: (B, S, N, C)

            # Compute correlation directly
            corrs = compute_corr_level(fmap1, fmap2s, C)
            corrs = corrs.reshape((B, S, N, H, W))

            # Prepare sampling grid:
            # Scale down the coordinates for the current level.
            centroid_lvl = coords.reshape((B * S * N, 1, 1, 2)) / (2**i)
            # Make sure our precomputed delta grid is on the same device/dtype.
            delta_lvl = self.delta.reshape((1, 2 * self.radius + 1, 2 * self.radius + 1, 2))
            # Now the grid for grid_sample is:
            # coords_lvl = centroid_lvl + delta_lvl   (broadcasted over grid)

            coords_lvl = centroid_lvl.broadcast_to((centroid_lvl.shape[0],  2 * self.radius + 1, 2 * self.radius + 1, 2)) + delta_lvl.broadcast_to((centroid_lvl.shape[0],  2 * self.radius + 1, 2 * self.radius + 1, 2))

            # Sample from the correlation volume using bilinear interpolation.
            # We reshape corrs to (B * S * N, 1, H, W) so grid_sample acts over each target.
            corrs_sampled = Tensor(bilinear_sampler(
                torch.from_numpy(corrs.numpy().reshape(B * S * N, 1, H, W)), torch.from_numpy(coords_lvl.numpy()), padding_mode=self.padding_mode
            ).numpy())
            # The sampled output is (B * S * N, 1, 2r+1, 2r+1). Flatten the last two dims.
            corrs_sampled = corrs_sampled.reshape((B, S, N, -1))  # Now shape: (B, S, N, (2r+1)^2)
            out_pyramid.append(corrs_sampled)

        # Concatenate all levels along the last dimension.
        out = ops.cat(out_pyramid, dim=-1)+0.0
        return out


def compute_corr_level(fmap1, fmap2s, C):
    # fmap1: (B, S, N, C)
    # fmap2s: (B, S, C, H*W)
    corrs = ops.matmul(fmap1, fmap2s)  # (B, S, N, H*W)
    corrs = corrs.reshape((fmap1.shape[0], fmap1.shape[1], fmap1.shape[2], -1))  # (B, S, N, H*W)
    return corrs / math.sqrt(C)