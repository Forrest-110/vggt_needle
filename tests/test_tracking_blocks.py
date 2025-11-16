#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#!/usr/bin/env python3
import numpy as np
import math
import torch

from needle import Tensor

# 🔧 adjust this to your actual file path where EfficientUpdateFormer / CorrBlock live
# e.g. from vggt.heads.track_modules.update_former import EfficientUpdateFormer, CorrBlock
from vggt.heads.track_modules.blocks import EfficientUpdateFormer, CorrBlock


def test_efficient_update_former_forward(add_space_attn: bool = True):
    """
    Test EfficientUpdateFormer forward pass.

    Expected input shape:
      input_tensor: (B, N, T, input_dim)
    Output:
      flow: (B, N_tracks, T, output_dim)
    """
    print(f"Testing EfficientUpdateFormer forward (add_space_attn={add_space_attn})...")

    B = 2
    N = 16
    T = 5
    input_dim = 320
    hidden_size = 384
    num_heads = 8
    output_dim = 130

    # Dummy input
    x_np = np.random.randn(B, N, T, input_dim).astype("float32")
    x = Tensor(x_np)

    model = EfficientUpdateFormer(
        space_depth=2,
        time_depth=6,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_heads=num_heads,
        output_dim=output_dim,
        add_space_attn=add_space_attn,
        num_virtual_tracks=4,  # keep small for test
    )

    flow, aux = model(x, mask=None)

    flow_np = flow.numpy()

    # If add_space_attn=True, model strips off virtual tracks at the end
    if add_space_attn:
        expected_N_tracks = N
    else:
        expected_N_tracks = N

    assert flow_np.shape[0] == B, f"B mismatch: {flow_np.shape}"
    assert flow_np.shape[1] == expected_N_tracks, f"N_tracks mismatch: {flow_np.shape}"
    assert flow_np.shape[2] == T, f"T mismatch: {flow_np.shape}"
    assert flow_np.shape[3] == output_dim, f"output_dim mismatch: {flow_np.shape}"
    assert np.isfinite(flow_np).all()
    print("  ✅ EfficientUpdateFormer forward OK; output shape:", flow_np.shape)


def test_corr_block_forward():
    """
    Test CorrBlock.corr_sample forward pass.

    fmaps: (B, S, C, H, W)
    targets: (B, S, N, C)
    coords: (B, S, N, 2)

    Output:
      corr: (B, S, N, L),
      where L = num_levels * (2*radius+1)^2
    """
    print("Testing CorrBlock forward...")

    B = 2
    S = 3
    C = 32
    H = 32
    W = 32
    N = 10
    num_levels = 3
    radius = 2  # (2r+1)^2 = 25

    # Dummy feature maps
    fmaps_np = np.random.randn(B, S, C, H, W).astype("float32")
    fmaps = Tensor(fmaps_np)

    corr_block = CorrBlock(
        fmaps=fmaps,
        num_levels=num_levels,
        radius=radius,
        multiple_track_feats=False,
        padding_mode="zeros",
    )

    # Dummy targets and coords
    # targets: (B, S, N, C) – use same C as fmaps
    targets_np = np.random.randn(B, S, N, C).astype("float32")
    targets = Tensor(targets_np)

    # coords: pixel coordinates at full resolution, within [0, H) and [0, W)
    coords_np = np.zeros((B, S, N, 2), dtype="float32")
    coords_np[..., 0] = np.random.uniform(0, H - 1, size=(B, S, N))  # y
    coords_np[..., 1] = np.random.uniform(0, W - 1, size=(B, S, N))  # x
    coords = Tensor(coords_np)

    corr = corr_block.corr_sample(targets, coords)
    corr_np = corr.numpy()

    L = num_levels * (2 * radius + 1) ** 2
    assert np.isfinite(corr_np).all()
    assert corr_np.shape == (B, S, N, L), f"CorrBlock output shape mismatch: {corr_np.shape}, expected {(B, S, N, L)}"

    print("  ✅ CorrBlock forward OK; output shape:", corr_np.shape)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    test_efficient_update_former_forward(add_space_attn=True)
    test_efficient_update_former_forward(add_space_attn=False)
    test_corr_block_forward()

    print("\nAll EfficientUpdateFormer / CorrBlock forward tests passed ✅")


if __name__ == "__main__":
    main()
