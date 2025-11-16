#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import torch

# 🔧 Adjust these imports to your project layout:
# e.g. from needle.nn import GroupNorm
from needle import Tensor
from needle.nn import GroupNorm


def copy_torch_groupnorm_to_needle(torch_gn: torch.nn.GroupNorm, needle_gn: GroupNorm):
    """
    Copy parameters from torch.nn.GroupNorm to Needle GroupNorm.
    Assumes:
      - needle_gn.weight, needle_gn.bias are shape (C,) if affine=True
    """
    if not torch_gn.affine:
        return

    with torch.no_grad():
        w = torch_gn.weight.detach().cpu().numpy()  # (C,)
        b = torch_gn.bias.detach().cpu().numpy()    # (C,)

    needle_gn.weight = Tensor(w.astype("float32"))
    needle_gn.bias = Tensor(b.astype("float32"))


def needle_to_numpy(x: Tensor) -> np.ndarray:
    return x.numpy()


def run_single_groupnorm_test(
    N: int,
    C: int,
    spatial_shape,
    num_groups: int,
    eps: float,
    affine: bool,
):
    """
    spatial_shape: tuple, e.g. (L,) or (H, W)
    """
    print(
        f"Testing GroupNorm: N={N}, C={C}, spatial={spatial_shape}, "
        f"num_groups={num_groups}, eps={eps}, affine={affine}"
    )

    torch.manual_seed(0)
    np.random.seed(0)

    # ----- PyTorch GroupNorm -----
    torch_gn = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=C,
        eps=eps,
        affine=affine,
    )

    # ----- Needle GroupNorm -----
    needle_gn = GroupNorm(
        num_groups=num_groups,
        num_channels=C,
        eps=eps,
        affine=affine,
        dtype="float32",
    )

    # Copy parameters
    copy_torch_groupnorm_to_needle(torch_gn, needle_gn)

    # ----- Input -----
    x_shape = (N, C, *spatial_shape)
    x_t = torch.randn(*x_shape, dtype=torch.float32)
    x_n = Tensor(x_t.detach().cpu().numpy())

    # ----- Forward -----
    y_t = torch_gn(x_t).detach().cpu().numpy()
    y_n = needle_to_numpy(needle_gn(x_n))

    # ----- Compare -----
    assert y_n.shape == y_t.shape, f"Shape mismatch: needle {y_n.shape}, torch {y_t.shape}"
    np.testing.assert_allclose(
        y_n,
        y_t,
        rtol=1e-5,
        atol=1e-6,
        err_msg="GroupNorm outputs mismatch",
    )
    print("  ✅ passed")


def test_groupnorm_all():
    # 1D spatial: (N, C, L)
    run_single_groupnorm_test(
        N=2,
        C=4,
        spatial_shape=(16,),
        num_groups=2,
        eps=1e-5,
        affine=True,
    )
    run_single_groupnorm_test(
        N=2,
        C=8,
        spatial_shape=(7,),
        num_groups=4,
        eps=1e-5,
        affine=False,
    )

    # 2D spatial: (N, C, H, W)
    run_single_groupnorm_test(
        N=2,
        C=8,
        spatial_shape=(8, 8),
        num_groups=4,
        eps=1e-5,
        affine=True,
    )
    run_single_groupnorm_test(
        N=1,
        C=16,
        spatial_shape=(5, 7),
        num_groups=8,
        eps=1e-5,
        affine=False,
    )


if __name__ == "__main__":
    test_groupnorm_all()
    print("\nAll GroupNorm alignment tests passed ✅")
