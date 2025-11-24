# #!/usr/bin/env python3
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# import numpy as np

# from needle import Tensor

# from vggt.heads.dpt_head import DPTHead


# def make_dummy_inputs(
#     B: int = 2,
#     S: int = 4,
#     H: int = 112,
#     W: int = 112,
#     patch_size: int = 14,
#     dim_in: int = 256,
#     patch_start_idx: int = 1,
#     intermediate_layer_idx = [4, 11, 17, 23],
# ):
#     """
#     Create dummy images + aggregated_tokens_list with correct shapes
#     for DPTHead forward.
#     """
#     # Images: [B, S, 3, H, W]
#     images_np = np.random.randn(B, S, 3, H, W).astype("float32")
#     images = Tensor(images_np)

#     # Token grid dimensions
#     patch_h, patch_w = H // patch_size, W // patch_size
#     num_patches = patch_h * patch_w
#     num_tokens_total = patch_start_idx + num_patches

#     max_layer_idx = max(intermediate_layer_idx)

#     aggregated_tokens_list = []
#     for _ in range(max_layer_idx + 1):
#         # Shape: [B, S, num_tokens_total, dim_in]
#         tokens_np = np.random.randn(B, S, num_tokens_total, dim_in).astype("float32")
#         aggregated_tokens_list.append(Tensor(tokens_np))

#     return aggregated_tokens_list, images, patch_start_idx, (patch_h, patch_w)


# def test_dpt_head_forward_pred_conf():
#     """
#     Test DPTHead forward in normal mode (feature_only=False):
#     - Check that forward runs without error
#     - Check that output shapes look sane
#     """
#     B, S, H, W = 2, 4, 112, 112
#     patch_size = 14
#     dim_in = 256
#     output_dim = 4

#     aggregated_tokens_list, images, patch_start_idx, _ = make_dummy_inputs(
#         B=B,
#         S=S,
#         H=H,
#         W=W,
#         patch_size=patch_size,
#         dim_in=dim_in,
#     )

#     head = DPTHead(
#         dim_in=dim_in,
#         patch_size=patch_size,
#         output_dim=output_dim,
#         feature_only=False,
#         pos_embed=True,
#         down_ratio=1,
#     )

#     # frames_chunk_size defaults to 8; since S=4 <= 8, it will process all frames at once (no chunking)
#     preds, conf = head(aggregated_tokens_list, images, patch_start_idx)

#     print("✅ DPTHead forward (feature_only=False) passed shape checks.")


# def test_dpt_head_forward_feature_only():
#     """
#     Test DPTHead forward in feature-only mode (feature_only=True):
#     - Returns feature maps instead of (preds, conf).
#     """
#     B, S, H, W = 1, 1, 112, 112
#     patch_size = 14
#     dim_in = 32
#     features = 32

#     aggregated_tokens_list, images, patch_start_idx, _ = make_dummy_inputs(
#         B=B,
#         S=S,
#         H=H,
#         W=W,
#         patch_size=patch_size,
#         dim_in=dim_in,
#     )

#     head = DPTHead(
#         dim_in=dim_in,
#         patch_size=patch_size,
#         output_dim=4,
#         feature_only=True,
#         features=features,
#         pos_embed=True,
#         down_ratio=1,
#     )

#     feats = head(aggregated_tokens_list, images, patch_start_idx)

#     # feats: [B, S, C, H, W]  with C == `features` (256)
#     assert feats.shape[0] == B, f"feats batch dim mismatch: {feats.shape}"
#     assert feats.shape[1] == S, f"feats seq dim mismatch: {feats.shape}"
#     assert feats.shape[2] == features, f"feats channel dim mismatch: {feats.shape}"
#     assert feats.shape[-2] == H, f"feats height mismatch: {feats.shape}"
#     assert feats.shape[-1] == W, f"feats width mismatch: {feats.shape}"

#     print("✅ DPTHead forward (feature_only=True) passed shape checks.")


# if __name__ == "__main__":
#     np.random.seed(0)
#     test_dpt_head_forward_pred_conf()
#     test_dpt_head_forward_feature_only()
#     print("All DPTHead forward tests passed ✅")


#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import needle as ndl
from needle import Tensor
from vggt.heads.dpt_head import DPTHead


def make_dummy_inputs(
    B: int = 2,
    S: int = 4,
    H: int = 112,
    W: int = 112,
    patch_size: int = 14,
    dim_in: int = 256,
    patch_start_idx: int = 1,
    intermediate_layer_idx = [4, 11, 17, 23],
    device=None,
):
    """
    Create dummy images + aggregated_tokens_list with correct shapes
    for DPTHead forward.
    """
    if device is None:
        device = ndl.cpu()

    # Images: [B, S, 3, H, W]
    images_np = np.random.randn(B, S, 3, H, W).astype("float32")
    images = Tensor(images_np, device=device)

    # Token grid dimensions
    patch_h, patch_w = H // patch_size, W // patch_size
    num_patches = patch_h * patch_w
    num_tokens_total = patch_start_idx + num_patches

    max_layer_idx = max(intermediate_layer_idx)

    aggregated_tokens_list = []
    for _ in range(max_layer_idx + 1):
        # Shape: [B, S, num_tokens_total, dim_in]
        tokens_np = np.random.randn(B, S, num_tokens_total, dim_in).astype("float32")
        aggregated_tokens_list.append(Tensor(tokens_np, device=device))

    return aggregated_tokens_list, images, patch_start_idx, (patch_h, patch_w)


def test_dpt_head_forward_pred_conf(device):
    """
    Test DPTHead forward in normal mode (feature_only=False):
    - Check that forward runs without error
    - Check that output shapes look sane
    """
    B, S, H, W = 2, 4, 112, 112
    patch_size = 14
    dim_in = 256
    output_dim = 4

    aggregated_tokens_list, images, patch_start_idx, _ = make_dummy_inputs(
        B=B,
        S=S,
        H=H,
        W=W,
        patch_size=patch_size,
        dim_in=dim_in,
        device=device,
    )

    head = DPTHead(
        dim_in=dim_in,
        patch_size=patch_size,
        output_dim=output_dim,
        feature_only=False,
        pos_embed=True,
        down_ratio=1,
    ).to(device)

    # frames_chunk_size defaults to 8; since S=4 <= 8, it will process all frames at once (no chunking)
    preds, conf = head(aggregated_tokens_list, images, patch_start_idx)

    # basic sanity checks
    assert preds.shape[0] == B, f"preds batch dim mismatch: {preds.shape}"
    assert preds.shape[1] == S, f"preds seq dim mismatch: {preds.shape}"
    assert conf.shape[0] == B and conf.shape[1] == S, f"conf shape mismatch: {conf.shape}"

    print("✅ DPTHead forward (feature_only=False) passed shape checks.")


def test_dpt_head_forward_feature_only(device):
    """
    Test DPTHead forward in feature-only mode (feature_only=True):
    - Returns feature maps instead of (preds, conf).
    """
    B, S, H, W = 1, 1, 112, 112
    patch_size = 14
    dim_in = 32
    features = 32

    aggregated_tokens_list, images, patch_start_idx, _ = make_dummy_inputs(
        B=B,
        S=S,
        H=H,
        W=W,
        patch_size=patch_size,
        dim_in=dim_in,
        device=device,
    )

    head = DPTHead(
        dim_in=dim_in,
        patch_size=patch_size,
        output_dim=4,
        feature_only=True,
        features=features,
        pos_embed=True,
        down_ratio=1,
    ).to(device)

    feats = head(aggregated_tokens_list, images, patch_start_idx)

    # feats: [B, S, C, H, W]  with C == `features`
    assert feats.shape[0] == B, f"feats batch dim mismatch: {feats.shape}"
    assert feats.shape[1] == S, f"feats seq dim mismatch: {feats.shape}"
    assert feats.shape[2] == features, f"feats channel dim mismatch: {feats.shape}"
    assert feats.shape[-2] == H, f"feats height mismatch: {feats.shape}"
    assert feats.shape[-1] == W, f"feats width mismatch: {feats.shape}"

    print("✅ DPTHead forward (feature_only=True) passed shape checks.")


if __name__ == "__main__":
    # Use CUDA device for Needle
    device = ndl.cuda()
    print(f"Running DPTHead tests on device: {device}")

    np.random.seed(0)
    test_dpt_head_forward_pred_conf(device)
    test_dpt_head_forward_feature_only(device)
    print("All DPTHead forward tests passed ✅")
