# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from needle import nn, Tensor, ops, init


def position_grid_to_embed(pos_grid: Tensor, embed_dim: int, omega_0: float = 100) -> Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape((-1, grid_dim))  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = ops.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.reshape((H, W, embed_dim))  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: Tensor, omega_0: float = 100) -> Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2

    device = pos.device
    omega = init.arange(embed_dim // 2, device=device)
    omega /= embed_dim / 2.0
    base =  init.constant(*(omega.shape), c=omega_0, device=device)
    omega = (base**omega)**-1  # (D/2,)

    pos = (pos+0.0).reshape((-1,))  # (M,)
    pos_exp = ops.reshape(pos, (pos.shape[0], 1)).broadcast_to((pos.shape[0], half))
    omega_exp = ops.reshape(omega, (1, half)).broadcast_to((pos.shape[0], half))
    out = pos_exp * omega_exp  # (M, D/2)

    emb_sin = ops.sin(out)  # (M, D/2)
    emb_cos = ops.cos(out)  # (M, D/2)

    emb = ops.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


# Inspired by https://github.com/microsoft/moge

def needle_linspace(start: float, end: float, steps: int, device=None, dtype="float32"):
    """
    Needle version of torch.linspace(start, end, steps).
    Returns shape (steps,)
    """
    if steps == 1:
        out = Tensor([start], device=device, dtype=dtype)
        return out

    idx = init.arange(steps, device=device, dtype=dtype)  # (steps,)
    # t = idx / (steps - 1)
    t = idx / float(steps - 1)
    # start + t * (end - start)
    out = start + t * (end - start)
    return out


def needle_meshgrid(x: Tensor, y: Tensor, indexing: str = "xy"):
    """
    Needle version of torch.meshgrid(x, y, indexing='xy' or 'ij').

    Args:
        x: (Nx,)
        y: (Ny,)

    Returns:
        If indexing == 'ij':
            xx, yy both (Nx, Ny)
        If indexing == 'xy':
            xx, yy both (Ny, Nx)
    """
    Nx = x.shape[0]
    Ny = y.shape[0]

    if indexing == "ij":
        # xx[i, j] = x[i], yy[i, j] = y[j]
        xx = ops.reshape(x, (Nx, 1))
        xx = ops.broadcast_to(xx, (Nx, Ny))

        yy = ops.reshape(y, (1, Ny))
        yy = ops.broadcast_to(yy, (Nx, Ny))
        return xx, yy

    elif indexing == "xy":
        # xx[i, j] = x[j], yy[i, j] = y[i]
        # first dim corresponds to y, second to x → (Ny, Nx)
        xx = ops.reshape(x, (1, Nx))
        xx = ops.broadcast_to(xx, (Ny, Nx))

        yy = ops.reshape(y, (Ny, 1))
        yy = ops.broadcast_to(yy, (Ny, Nx))
        return xx, yy

    else:
        raise ValueError("indexing must be 'xy' or 'ij'")
    
def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None,
    dtype="float32", device=None
) -> Tensor:
    """
    Needle version of create_uv_grid.
    Returns (width, height, 2)
    """
    # ---------------------------------
    # aspect ratio + spans
    # ---------------------------------
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # ---------------------------------
    # linspace for x and y
    # ---------------------------------
    x_coords = needle_linspace(left_x, right_x, width, device=device, dtype=dtype)
    y_coords = needle_linspace(top_y, bottom_y, height, device=device, dtype=dtype)

    # ---------------------------------
    # meshgrid (W,H)
    # ---------------------------------
    uu, vv = needle_meshgrid(x_coords, y_coords, indexing="xy")

    # ---------------------------------
    # stack -> (W, H, 2)
    # ---------------------------------
    uv = ops.stack([uu, vv], axis=-1)

    return uv