#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from needle import Tensor

# 🔧 Adjust if VGGT lives somewhere else, e.g. vggt.models.vggt_model
from vggt.models.vggt import VGGT


def test_vggt_forward_with_tracking():
    """
    Test VGGT forward pass with all heads enabled and query_points provided.
    """
    print("Testing VGGT forward (with tracking)...")

    # ------------------------------------------------------------------
    # Dummy sizes
    # ------------------------------------------------------------------
    B = 1          # batch size
    S = 2          # sequence length
    H = 518        # must match img_size for default Aggregator
    W = 518
    C = 3          # RGB
    N = 5          # number of query points
    img_size = 518
    patch_size = 14
    embed_dim = 384  # default

    # ------------------------------------------------------------------
    # Dummy images: (B, S, 3, H, W)
    # ------------------------------------------------------------------
    images_np = np.random.rand(B, S, C, H, W).astype("float32")
    images = Tensor(images_np)

    # ------------------------------------------------------------------
    # Dummy query points: (B, N, 2) in pixel coordinates
    # ------------------------------------------------------------------
    query_np = np.zeros((B, N, 2), dtype="float32")
    query_np[..., 0] = np.random.uniform(0, W - 1, size=(B, N))  # x
    query_np[..., 1] = np.random.uniform(0, H - 1, size=(B, N))  # y
    query_points = Tensor(query_np)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
    )
    model.eval()  # so that predictions["images"] is filled

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    preds = model(images, query_points=query_points)

    # ------------------------------------------------------------------
    # Basic key checks
    # ------------------------------------------------------------------
    expected_keys = [
        "pose_enc",
        "pose_enc_list",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "track",
        "vis",
        "conf",
        "images",
    ]
    for k in expected_keys:
        assert k in preds, f"Missing key in predictions: {k}"

    # ------------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------------
    pose_enc = preds["pose_enc"]
    assert pose_enc.shape[0] == B and pose_enc.shape[1] == S, \
        f"pose_enc shape mismatch: {pose_enc.shape}"

    depth = preds["depth"]
    depth_conf = preds["depth_conf"]
    wp = preds["world_points"]
    wp_conf = preds["world_points_conf"]
    track = preds["track"]
    vis = preds["vis"]
    conf = preds["conf"]

    # depth, depth_conf, world_points, world_points_conf: expect (B, S, ..., H', W' or H, W)
    # We don't enforce channel dims here, just batch/seq and spatial shapes
    assert depth.shape[0] == B and depth.shape[1] == S, f"depth shape mismatch: {depth.shape}"
    assert depth_conf.shape[0] == B and depth_conf.shape[1] == S, f"depth_conf shape mismatch: {depth_conf.shape}"
    assert wp.shape[0] == B and wp.shape[1] == S, f"world_points shape mismatch: {wp.shape}"
    assert wp_conf.shape[0] == B and wp_conf.shape[1] == S, f"world_points_conf shape mismatch: {wp_conf.shape}"

    # track: (B, S, N, 2)
    assert track.shape[0] == B and track.shape[1] == S and track.shape[2] == N and track.shape[3] == 2, \
        f"track shape mismatch: {track.shape}"

    # vis & conf: (B, S, N)
    assert vis.shape == (B, S, N), f"vis shape mismatch: {vis.shape}"
    assert conf.shape == (B, S, N), f"conf shape mismatch: {conf.shape}"

    # images key: (B, S, 3, H, W)
    out_images = preds["images"]
    assert out_images.shape == (B, S, C, H, W), \
        f"images shape mismatch: {out_images.shape}, expected {(B, S, C, H, W)}"

    print("  ✓ VGGT forward with tracking passed basic shape checks.")


def test_vggt_forward_no_tracking():
    """
    Test VGGT forward pass with query_points=None (no tracking output).
    """
    print("Testing VGGT forward (no tracking)...")

    B = 1
    S = 2
    H = 518
    W = 518
    C = 3
    img_size = 518
    patch_size = 14
    embed_dim = 1024

    images_np = np.random.rand(B, S, C, H, W).astype("float32")
    images = Tensor(images_np)

    model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,  # track_head exists but won't be used without query_points
    )
    model.eval()

    preds = model(images, query_points=None)

    # Keys that should exist
    must_have = [
        "pose_enc",
        "pose_enc_list",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
    ]
    for k in must_have:
        assert k in preds, f"Missing key in predictions: {k}"

    # Tracking keys should NOT be present
    for k in ["track", "vis", "conf"]:
        assert k not in preds, f"Key {k} should not be present when query_points is None"

    print("  ✓ VGGT forward without tracking passed key checks.")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    test_vggt_forward_with_tracking()
    test_vggt_forward_no_tracking()

    print("\nAll VGGT forward tests passed ✅")
