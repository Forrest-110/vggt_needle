# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from needle import ops, Tensor

def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[:, :, :3]
    quat = pred_pose_enc[:, :, 3:7]
    fl = pred_pose_enc[:, :, 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = ops.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc: Tensor, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return ops.exp(pose_enc)
    elif act_type == "relu":
        return ops.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out: Tensor, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute((0, 2, 3, 1))  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = (fmap[:, :, :, -1]+0.0).reshape((fmap.shape[0], fmap.shape[1], fmap.shape[2]))
   
    if activation == "norm_exp":
        d = ops.clamp(ops.norm(xyz, dim=-1, keepdim=True).broadcast_to(xyz.shape), min_val=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * (ops.exp(d) - 1)
    elif activation == "norm":
        pts3d = xyz / ops.norm(xyz, dim=-1, keepdim=True).broadcast_to(xyz.shape)
    elif activation == "exp":
        pts3d = ops.exp(xyz)
    elif activation == "relu":
        pts3d = ops.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        raise NotImplementedError
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = ops.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        raise NotImplementedError
        pts3d = ops.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + ops.exp(conf)
    elif conf_activation == "expp0":
        conf_out = ops.exp(conf)
    elif conf_activation == "sigmoid":
        raise NotImplementedError
        conf_out = ops.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out

def abs_tensor(x):
    return ops.where(x >= 0, x, -x)

def inverse_log_transform(y):
    abs_y = ops.abs(y)
    expm1_y = ops.exp(abs_y) - 1
    sign_y = ops.sign(y)
    return sign_y * expm1_y