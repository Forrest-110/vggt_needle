#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

from needle import Tensor

from vggt.heads.track_modules.base_track_predictor import BaseTrackerPredictor


def make_dummy_inputs(
    B: int = 2,
    N: int = 5,
    S: int = 4,
    latent_dim: int = 128,
    HH: int = 16,
    WW: int = 16,
):
    """
    Create dummy query_points and fmaps with the right shapes:

      query_points: (B, N, 2)
      fmaps:        (B, S, C, HH, WW), with C = latent_dim
    """
    C = latent_dim

    # Query points: random coordinates in [0, min(HH, WW))
    coords_np = np.zeros((B, N, 2), dtype="float32")
    coords_np[..., 0] = np.random.uniform(0, HH - 1, size=(B, N))  # y or x, not super important
    coords_np[..., 1] = np.random.uniform(0, WW - 1, size=(B, N))

    query_points = Tensor(coords_np)

    # Feature maps: random latent-dim channels
    fmaps_np = np.random.randn(B, S, C, HH, WW).astype("float32")
    fmaps = Tensor(fmaps_np)

    return query_points, fmaps


def test_base_tracker_forward_basic():
    """
    Test forward pass of BaseTrackerPredictor with default settings
    and a small number of iterations.
    """
    print("Testing BaseTrackerPredictor forward (return_feat=False)...")

    B, N, S = 2, 5, 4
    latent_dim = 128
    HH, WW = 16, 16
    iters = 2

    query_points, fmaps = make_dummy_inputs(
        B=B, N=N, S=S, latent_dim=latent_dim, HH=HH, WW=WW
    )

    model = BaseTrackerPredictor(
        stride=1,
        corr_levels=3,
        corr_radius=2,
        latent_dim=latent_dim,
        hidden_size=384,
        use_spaceatt=True,
        depth=3,
        max_scale=518,
        predict_conf=True,
    )

    coord_preds, vis_e, conf_e = model(
        query_points=query_points,
        fmaps=fmaps,
        iters=iters,
        return_feat=False,
        down_ratio=1,
        apply_sigmoid=True,
    )

    # coord_preds: list of length == iters, each (B, S, N, 2)
    assert isinstance(coord_preds, list), "coord_preds should be a list"
    assert len(coord_preds) == iters, f"coord_preds length {len(coord_preds)} != iters {iters}"

    for i, coords_i in enumerate(coord_preds):
        coords_np = coords_i.numpy()
        assert coords_np.shape == (B, S, N, 2), \
            f"coord_preds[{i}] shape mismatch: {coords_np.shape}, expected {(B, S, N, 2)}"

    # vis_e: (B, S, N)
    vis_np = vis_e.numpy()
    assert vis_np.shape == (B, S, N), f"vis_e shape mismatch: {vis_np.shape}, expected {(B, S, N)}"
    # with sigmoid -> in (0,1)
    assert (vis_np >= 0).all() and (vis_np <= 1).all(), f"vis_e values should be in [0,1] when apply_sigmoid=True, get min {vis_np.min()}, max {vis_np.max()}"

    # conf_e: (B, S, N)
    assert conf_e is not None, "conf_e should not be None when predict_conf=True"
    conf_np = conf_e.numpy()
    assert conf_np.shape == (B, S, N), f"conf_e shape mismatch: {conf_np.shape}, expected {(B, S, N)}"
    assert (conf_np >= 0).all() and (conf_np <= 1).all(), "conf_e values should be in [0,1] when apply_sigmoid=True"

    print("  ✅ forward (return_feat=False) passed shape & range checks.")


def test_base_tracker_forward_with_feats():
    """
    Test forward pass of BaseTrackerPredictor with return_feat=True.
    """
    print("Testing BaseTrackerPredictor forward (return_feat=True)...")

    B, N, S = 2, 3, 3
    latent_dim = 128
    HH, WW = 8, 8
    iters = 1

    query_points, fmaps = make_dummy_inputs(
        B=B, N=N, S=S, latent_dim=latent_dim, HH=HH, WW=WW
    )

    model = BaseTrackerPredictor(
        stride=1,
        corr_levels=2,
        corr_radius=1,
        latent_dim=latent_dim,
        hidden_size=256,
        use_spaceatt=False,  # also test no space attention
        depth=2,
        max_scale=256,
        predict_conf=False,  # also test predict_conf=False
    )

    coord_preds, vis_e, track_feats, query_track_feat, conf_e = model(
        query_points=query_points,
        fmaps=fmaps,
        iters=iters,
        return_feat=True,
        down_ratio=1,
        apply_sigmoid=False,
    )

    # coord_preds: list length iters, each (B,S,N,2)
    assert isinstance(coord_preds, list)
    assert len(coord_preds) == iters
    coords_np = coord_preds[0].numpy()
    assert coords_np.shape == (B, S, N, 2), \
        f"coord_preds[0] shape mismatch: {coords_np.shape}, expected {(B, S, N, 2)}"

    # vis_e: (B,S,N)
    vis_np = vis_e.numpy()
    assert vis_np.shape == (B, S, N), f"vis_e shape mismatch: {vis_np.shape}, expected {(B, S, N)}"

    # track_feats: expected (B,S,N,latent_dim)
    tf_np = track_feats.numpy()
    assert tf_np.shape == (B, S, N, latent_dim), \
        f"track_feats shape mismatch: {tf_np.shape}, expected {(B, S, N, latent_dim)}"

    # query_track_feat: PyTorch tensor from sample_features4d, shape (B,N,C)
    assert isinstance(query_track_feat, torch.Tensor), "query_track_feat should be a torch.Tensor"
    assert query_track_feat.shape == (B, N, latent_dim), \
        f"query_track_feat shape mismatch: {query_track_feat.shape}, expected {(B, N, latent_dim)}"

    # conf_e: should be None when predict_conf=False
    assert conf_e is None, "conf_e should be None when predict_conf=False"

    print("  ✅ forward (return_feat=True) passed shape checks.")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    test_base_tracker_forward_basic()
    test_base_tracker_forward_with_feats()

    print("\nAll BaseTrackerPredictor forward tests passed ✅")
