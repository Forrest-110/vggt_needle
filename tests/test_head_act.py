#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


"""
Test alignment between Needle pose/activation functions and PyTorch reference:

- inverse_log_transform
- base_pose_act / activate_pose
- activate_head

We only test the branches that are currently implemented in the Needle code:
    base_pose_act: linear, inv_log, exp, relu
    activate_head.activation: norm_exp, norm, exp, relu, inv_log, linear
    activate_head.conf_activation: expp1, expp0
"""

import numpy as np
import torch

from needle import Tensor, ops, init


from vggt.heads.head_act import (
    inverse_log_transform,
    base_pose_act,
    activate_pose,
    activate_head,
)


# -----------------------------
# PyTorch reference functions
# -----------------------------

def torch_inverse_log_transform(y: torch.Tensor) -> torch.Tensor:
    """
    Reference: sign(y) * (expm1(|y|))
    """
    return torch.sign(y) * torch.expm1(torch.abs(y))


def torch_base_pose_act(pose_enc: torch.Tensor, act_type: str = "linear") -> torch.Tensor:
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return torch_inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return torch.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def torch_activate_pose(pred_pose_enc: torch.Tensor,
                        trans_act: str = "linear",
                        quat_act: str = "linear",
                        fl_act: str = "linear") -> torch.Tensor:
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]

    T = torch_base_pose_act(T, trans_act)
    quat = torch_base_pose_act(quat, quat_act)
    fl = torch_base_pose_act(fl, fl_act)

    return torch.cat([T, quat, fl], dim=-1)


def torch_activate_head(out: torch.Tensor,
                        activation: str = "norm_exp",
                        conf_activation: str = "expp1"):
    """
    out: (B, C, H, W)
    returns: (pts3d, conf_out)
    """
    # (B, C, H, W) -> (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)
    xyz = fmap[..., :-1]
    conf = fmap[..., -1]
    # print(fmap)
    if activation == "norm_exp":
        d = torch.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * (torch.exp(d) - 1)
    elif activation == "norm":
        pts3d = xyz / torch.norm(xyz, dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = torch.relu(xyz)
    elif activation == "inv_log":
        pts3d = torch_inverse_log_transform(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + torch.exp(conf)
    elif conf_activation == "expp0":
        conf_out = torch.exp(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")
    
    return pts3d, conf_out


# -----------------------------
# Utility: comparison helper
# -----------------------------

def compare_tensors(name: str, needle_tensor: Tensor, torch_tensor: torch.Tensor, atol=1e-5, rtol=1e-5):
    np_needle = needle_tensor.numpy()
    torch_from_needle = torch.from_numpy(np_needle).to(torch_tensor.dtype)
    
    diff = (torch_from_needle - torch_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"[{name}] max diff = {max_diff:.6e}, mean diff = {mean_diff:.6e}")
    ok = max_diff <= atol + rtol * torch_tensor.abs().max().item()
    print(f"  -> {'OK' if ok else 'MISMATCH'}")
    return ok


# -----------------------------
# Individual tests
# -----------------------------

def test_inverse_log_transform():
    print("\n=== test_inverse_log_transform ===")
    np.random.seed(0)

    x_np = np.random.randn(4, 5).astype("float32") * 3.0
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    x_needle = Tensor(x_np)

    y_torch = torch_inverse_log_transform(x_torch)
    y_needle = inverse_log_transform(x_needle)

    compare_tensors("inverse_log_transform", y_needle, y_torch)


def test_base_pose_and_activate_pose():
    print("\n=== test_base_pose_act / activate_pose ===")
    np.random.seed(1)

    # shape (..., 8): 3T + 4quat + 1fl
    x_np = np.random.randn(2, 3, 8).astype("float32")
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    x_needle = Tensor(x_np)

    act_types = ["linear", "inv_log", "exp", "relu"]

    # base_pose_act, per activation
    for act in act_types:
        print(f"\n-- base_pose_act act_type={act} --")
        t_ref = torch_base_pose_act(x_torch, act)
        t_needle = base_pose_act(x_needle, act)
        compare_tensors(f"base_pose_act[{act}]", t_needle, t_ref)

    # activate_pose – test some combinations
    combos = [
        ("linear", "linear", "linear"),
        ("inv_log", "linear", "linear"),
        ("linear", "inv_log", "exp"),
        ("relu", "relu", "relu"),
    ]
    for trans_act, quat_act, fl_act in combos:
        print(f"\n-- activate_pose trans={trans_act}, quat={quat_act}, fl={fl_act} --")
        y_ref = torch_activate_pose(x_torch, trans_act=trans_act, quat_act=quat_act, fl_act=fl_act)
        y_needle = activate_pose(x_needle, trans_act=trans_act, quat_act=quat_act, fl_act=fl_act)
        compare_tensors(f"activate_pose[{trans_act},{quat_act},{fl_act}]", y_needle, y_ref)


def test_activate_head():
    print("\n=== test_activate_head ===")
    np.random.seed(2)

    B, C, H, W = 2, 4, 5, 6  # C-1=3 for xyz, 1 for conf
    x_np = np.random.randn(B, C, H, W).astype("float32")
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    x_needle = Tensor(x_np)

    activations = ["norm_exp", "norm", "exp", "relu", "inv_log", "linear"]
    conf_acts = ["expp1", "expp0"]

    for act in activations:
        for cact in conf_acts:
            print(f"\n-- activate_head activation={act}, conf_activation={cact} --")
            pts_ref, conf_ref = torch_activate_head(x_torch, activation=act, conf_activation=cact)
            pts_needle, conf_needle = activate_head(x_needle, activation=act, conf_activation=cact)
            
            ok1 = compare_tensors(f"activate_head.pts3d[{act},{cact}]", pts_needle, pts_ref)
            ok2 = compare_tensors(f"activate_head.conf[{act},{cact}]", conf_needle, conf_ref)
            # not asserting here, just printing – you can turn this into asserts if desired


def main():
    test_inverse_log_transform()
    test_base_pose_and_activate_pose()
    test_activate_head()
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
