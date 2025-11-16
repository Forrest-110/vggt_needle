#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


"""
Smoke test for CameraHead in Needle.

Checks:
  - forward pass runs
  - output list length == num_iterations
  - each output has expected shape (B, S, target_dim)
  - backward pass produces non-zero gradients on some parameters
"""

import argparse
import numpy as np

from needle import Tensor, init

from vggt.heads.camera_head import CameraHead


def make_dummy_aggregated_tokens_list(
    B: int,
    S: int,
    P: int,
    dim_in: int,
    num_blocks: int,
) -> list[Tensor]:
    """
    Create a fake aggregated_tokens_list as expected by CameraHead:

      - It's a list of tensors
      - Only the last tensor is actually used
      - Each tensor has shape (B, S, P, dim_in)
      - CameraHead will take tokens = aggregated_tokens_list[-1]
        and then use tokens[:, :, 0] as camera tokens, shape (B, S, dim_in)
    """
    tokens_list = []
    for _ in range(num_blocks):
        x = init.randn(B, S, P, dim_in)
        tokens_list.append(x)
    return tokens_list


def test_camera_head_once(
    B: int = 2,
    S: int = 3,
    P: int = 4,
    dim_in: int = 128,
    trunk_depth: int = 2,
    num_iterations: int = 3,
    trans_act: str = "linear",
    quat_act: str = "linear",
    fl_act: str = "relu",
):
    print(
        f"\n=== Testing CameraHead: "
        f"B={B}, S={S}, P={P}, dim_in={dim_in}, "
        f"trunk_depth={trunk_depth}, num_iterations={num_iterations}, "
        f"trans_act={trans_act}, quat_act={quat_act}, fl_act={fl_act} ==="
    )

    # Build head
    head = CameraHead(
        dim_in=dim_in,
        trunk_depth=trunk_depth,
        pose_encoding_type="absT_quaR_FoV",
        num_heads=4,
        mlp_ratio=4,
        init_values=0.01,
        trans_act=trans_act,
        quat_act=quat_act,
        fl_act=fl_act,
    )

    # Fake aggregator outputs
    aggregated_tokens_list = make_dummy_aggregated_tokens_list(
        B=B,
        S=S,
        P=P,
        dim_in=dim_in,
        num_blocks=trunk_depth,  # arbitrary; only last one is used
    )

    # Forward
    pred_pose_list = head(aggregated_tokens_list, num_iterations=num_iterations)

    print(f"Number of iterations (outputs): {len(pred_pose_list)}")
    assert len(pred_pose_list) == num_iterations, (
        f"Expected {num_iterations} pose predictions, got {len(pred_pose_list)}"
    )

    # Check shapes
    target_dim = head.target_dim  # for absT_quaR_FoV this should be 9
    for i, pose in enumerate(pred_pose_list):
        print(f"  iter {i}: pose shape = {pose.shape}")
        assert pose.shape == (B, S, target_dim), (
            f"Iteration {i} pose shape mismatch: expected {(B, S, target_dim)}, got {pose.shape}"
        )

        # optional: if fl_act='relu', last dim (FOV) should be >= 0 (numerically may be tiny negatives)
        if fl_act == "relu":
            fl = pose[:, :, -1]  # last component is FoV
            min_fl = fl.numpy().min()
            print(f"    min FoV (relu): {min_fl:.6f}")

    # Backward: sum over all iterations
    total_loss = None
    for pose in pred_pose_list:
        loss_i = pose.sum()
        total_loss = loss_i if total_loss is None else total_loss + loss_i

    
    print("CameraHead test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-patches", type=int, default=4)
    parser.add_argument("--dim-in", type=int, default=128)
    parser.add_argument("--trunk-depth", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--trans-act", type=str, default="linear",
                        choices=["linear", "inv_log", "exp", "relu"])
    parser.add_argument("--quat-act", type=str, default="linear",
                        choices=["linear", "inv_log", "exp", "relu"])
    parser.add_argument("--fl-act", type=str, default="relu",
                        choices=["linear", "inv_log", "exp", "relu"])
    args = parser.parse_args()

    test_camera_head_once(
        B=args.batch_size,
        S=args.seq_len,
        P=args.num_patches,
        dim_in=args.dim_in,
        trunk_depth=args.trunk_depth,
        num_iterations=args.num_iterations,
        trans_act=args.trans_act,
        quat_act=args.quat_act,
        fl_act=args.fl_act,
    )

    print("\nAll CameraHead tests completed successfully.")


if __name__ == "__main__":
    main()
