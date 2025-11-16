#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from needle import Tensor, ops, init

# -------------------------------------------------------------------
# IMPORT YOUR NEEDLE IMPLEMENTATIONS HERE
# -------------------------------------------------------------------
# Replace `your_module` with the actual filename (without .py)
# e.g. from pos_embed import ...
from vggt.heads.utils import (
    needle_linspace,
    needle_meshgrid,
    create_uv_grid,
    make_sincos_pos_embed,
    position_grid_to_embed,
)


# -------------------------------------------------------------------
# TORCH REFERENCE IMPLEMENTATIONS
# -------------------------------------------------------------------

def torch_make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    device = pos.device
    dtype = pos.dtype

    omega = torch.arange(half, device=device, dtype=dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = pos[:, None] * omega[None, :]  # (M, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (M, D)
    return emb


def torch_position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100.0) -> torch.Tensor:
    """
    Torch reference version of position_grid_to_embed.
    pos_grid: (H, W, 2)
    returns: (H, W, embed_dim)
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (H*W, 2)

    emb_x = torch_make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = torch_make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = torch.cat([emb_x, emb_y], dim=-1)  # (H*W, D)
    return emb.reshape(H, W, embed_dim)


def torch_create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Original torch implementation of create_uv_grid (reference).
    Returns (width, height, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


# -------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------

def needle_to_numpy(x: Tensor) -> np.ndarray:
    # Adjust this if your Tensor API is different
    return x.numpy()


def assert_allclose(a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-6, msg=""):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


# -------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------

def test_linspace():
    print("Testing needle_linspace vs torch.linspace...")
    cases = [
        (0.0, 1.0, 1),
        (0.0, 1.0, 2),
        (0.0, 1.0, 5),
        (-1.0, 3.0, 7),
        (-2.5, 2.5, 10),
    ]
    for start, end, steps in cases:
        needle_out = needle_linspace(start, end, steps, dtype="float32")
        torch_out = torch.linspace(start, end, steps=steps, dtype=torch.float32)

        needle_np = needle_to_numpy(needle_out)
        torch_np = torch_out.cpu().numpy()

        assert needle_np.shape == torch_np.shape
        assert_allclose(needle_np, torch_np, msg=f"linspace mismatch for {start}, {end}, {steps}")

    print("  ✓ needle_linspace matches torch.linspace")


def test_meshgrid():
    print("Testing needle_meshgrid vs torch.meshgrid...")

    width, height = 7, 5
    # You can also generate from needle_linspace, but it's nice to test
    # against arbitrary coordinates as well.
    x = np.linspace(-1.0, 1.0, width).astype("float32")
    y = np.linspace(-0.5, 0.5, height).astype("float32")

    x_n = Tensor(x)
    y_n = Tensor(y)
    uu_n, vv_n = needle_meshgrid(x_n, y_n, indexing="xy")

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    uu_t, vv_t = torch.meshgrid(x_t, y_t, indexing="xy")

    uu_np = needle_to_numpy(uu_n)
    vv_np = needle_to_numpy(vv_n)

    assert_allclose(uu_np, uu_t.numpy(), msg="meshgrid uu mismatch")
    assert_allclose(vv_np, vv_t.numpy(), msg="meshgrid vv mismatch")

    print("  ✓ needle_meshgrid matches torch.meshgrid")


def test_create_uv_grid():
    print("Testing create_uv_grid vs torch_create_uv_grid...")

    for width, height in [(4, 3), (7, 5), (16, 9)]:
        aspect_ratio = float(width) / float(height)

        torch_uv = torch_create_uv_grid(width, height, aspect_ratio=aspect_ratio, dtype=torch.float32)
        needle_uv = create_uv_grid(width, height, aspect_ratio=aspect_ratio, dtype="float32")

        torch_np = torch_uv.numpy()
        needle_np = needle_to_numpy(needle_uv)
        assert torch_np.shape == needle_np.shape == (height, width, 2)
        assert_allclose(needle_np, torch_np, msg=f"uv grid mismatch for {width}x{height}")

    print("  ✓ create_uv_grid matches torch_create_uv_grid")


def test_make_sincos_pos_embed():
    print("Testing make_sincos_pos_embed vs torch_make_sincos_pos_embed...")

    embed_dim = 16
    positions = np.linspace(-1.0, 1.0, 13).astype("float32")

    pos_n = Tensor(positions)
    pos_t = torch.from_numpy(positions)

    needle_emb = make_sincos_pos_embed(embed_dim, pos_n, omega_0=100.0)
    torch_emb = torch_make_sincos_pos_embed(embed_dim, pos_t, omega_0=100.0)

    needle_np = needle_to_numpy(needle_emb)
    torch_np = torch_emb.detach().numpy()

    assert needle_np.shape == torch_np.shape == (positions.shape[0], embed_dim)
    assert_allclose(needle_np, torch_np, msg="make_sincos_pos_embed mismatch")

    print("  ✓ make_sincos_pos_embed matches torch_make_sincos_pos_embed")


def test_position_grid_to_embed():
    print("Testing position_grid_to_embed vs torch_position_grid_to_embed...")

    H, W = 5, 7
    embed_dim = 32

    # Use the Torch reference UV grid, but treat it as (H, W, 2).
    # Our torch_create_uv_grid returns (W, H, 2), so we transpose first.
    torch_uv_wh2 = torch_create_uv_grid(W, H, aspect_ratio=float(W) / float(H), dtype=torch.float32)
    # torch_uv_wh2: (W, H, 2)
    pos_grid_t = torch_uv_wh2.permute(1, 0, 2).contiguous()  # (H, W, 2)

    # Torch reference embedding
    emb_t = torch_position_grid_to_embed(pos_grid_t, embed_dim, omega_0=100.0)

    # Needle embedding (convert pos_grid_t to Needle Tensor)
    pos_grid_n = Tensor(pos_grid_t.numpy().astype("float32"))
    emb_n = position_grid_to_embed(pos_grid_n, embed_dim, omega_0=100.0)

    emb_t_np = emb_t.detach().numpy()
    emb_n_np = needle_to_numpy(emb_n)
    # print(emb_t_np.shape, emb_n_np.shape)
    assert emb_t_np.shape == emb_n_np.shape
    assert_allclose(emb_n_np, emb_t_np, msg="position_grid_to_embed mismatch")

    print("  ✓ position_grid_to_embed matches torch_position_grid_to_embed")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_linspace()
    test_meshgrid()
    test_create_uv_grid()
    test_make_sincos_pos_embed()
    test_position_grid_to_embed()
    print("\nAll tests passed ✅")
