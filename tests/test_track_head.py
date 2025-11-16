#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from needle import Tensor

# 🔧 FIX THESE IMPORTS if needed
from vggt.heads.track_head import TrackHead
from vggt.heads.dpt_head import DPTHead   # ensures module loads
from vggt.heads.track_modules.base_track_predictor import BaseTrackerPredictor


def test_track_head_forward():
    print("Testing TrackHead forward pass...")

    # ---------------------------------------------------------
    # Dummy sizes
    # ---------------------------------------------------------
    B = 2        # batch size
    S = 3        # sequence length
    H = 64       # image height
    W = 64       # image width
    C = 3        # RGB images
    dim_in = 128 # token dimension
    N = 4        # number of tracked points
    num_layers = 24 # length of aggregated tokens list
    patch_start_idx = 0

    # ---------------------------------------------------------
    # Create dummy aggregated_tokens_list
    # Each element: (B, S, num_tokens, dim_in)
    # Let’s use 16 patch tokens for simplicity
    num_patch_tokens = 16
    aggregated_tokens_list = []
    for _ in range(num_layers):
        arr = np.random.randn(B, S, num_patch_tokens, dim_in).astype("float32")
        aggregated_tokens_list.append(Tensor(arr))

    # ---------------------------------------------------------
    # Dummy images
    images_np = np.random.randn(B, S, C, H, W).astype("float32")
    images = Tensor(images_np)

    # ---------------------------------------------------------
    # Dummy query points (B, N, 2)
    query_np = np.random.uniform(low=0, high=min(H, W), size=(B, N, 2)).astype("float32")
    query_points = Tensor(query_np)

    # ---------------------------------------------------------
    # Build TrackHead
    track_head = TrackHead(
        dim_in=dim_in,
        patch_size=14,
        features=64,         # small feature dimension for faster test
        iters=2,             # few iterations, fast
        predict_conf=True,
        stride=2,
        corr_levels=3,
        corr_radius=2,
        hidden_size=128,
    )

    # ---------------------------------------------------------
    # Forward Pass
    coord_preds, vis_scores, conf_scores = track_head(
        aggregated_tokens_list=aggregated_tokens_list,
        images=images,
        patch_start_idx=patch_start_idx,
        query_points=query_points,
        iters=2,
    )

    # ---------------------------------------------------------
    # Assertions
    # coord_preds is a list of length = iters
    assert isinstance(coord_preds, list), "coord_preds must be a list"
    assert len(coord_preds) == 2, f"coord_preds length mismatch: {len(coord_preds)} != 2"

    for i, coords in enumerate(coord_preds):
        coords_np = coords.numpy()
        assert coords_np.shape == (B, S, N, 2), \
            f"coord_preds[{i}] shape mismatch: {coords_np.shape} != {(B, S, N, 2)}"

    # vis_scores shape: (B, S, N)
    vs_np = vis_scores.numpy()
    assert vs_np.shape == (B, S, N), \
        f"vis_scores shape mismatch: {vs_np.shape} != {(B, S, N)}"

    # conf_scores shape: (B, S, N)
    conf_np = conf_scores.numpy()
    assert conf_np.shape == (B, S, N), \
        f"conf_scores shape mismatch: {conf_np.shape} != {(B, S, N)}"

    print("  ✓ TrackHead forward pass OK")
    print("    coord_preds[0] shape:", coord_preds[0].shape)
    print("    vis_scores shape    :", vis_scores.shape)
    print("    conf_scores shape   :", conf_scores.shape)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    test_track_head_forward()
    print("\nAll tests passed! ✅")
