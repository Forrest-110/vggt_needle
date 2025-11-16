#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from needle import Tensor
from needle import nn
from vggt.layers.patch_embed import *

if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn_torch

    # ---------- Torch reference implementation ----------

    class TorchPatchEmbed(nn_torch.Module):
        def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            use_norm: bool = True,
            flatten_embedding: bool = True,
        ):
            super().__init__()
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)

            self.img_size = img_size
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.flatten_embedding = flatten_embedding

            self.proj = nn_torch.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.norm = nn_torch.LayerNorm(embed_dim) if use_norm else nn_torch.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            ph, pw = self.patch_size
            assert H % ph == 0 and W % pw == 0
            x = self.proj(x)  # (B, C', H', W')

            H_out, W_out = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)  # (B, H'W', C')
            x = self.norm(x)
            if not self.flatten_embedding:
                x = x.view(B, H_out, W_out, self.embed_dim)
            return x

    # ---------- Utility to align Torch & Needle params ----------

    def make_aligned_patch_embed(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        use_norm: bool,
        flatten_embedding: bool,
    ):
        # Torch module
        torch_pe = TorchPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_norm=use_norm,
            flatten_embedding=flatten_embedding,
        )

        # Needle module
        norm_layer = nn.LayerNorm if use_norm else None
        needle_pe = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten_embedding=flatten_embedding,
        )

        # Copy conv weights: Torch (O, I, H, W) -> Needle Conv expects HWIO
        w_torch = torch_pe.proj.weight.detach().cpu().numpy()        # (O, I, H, W)
        w_hwio = np.transpose(w_torch, (2, 3, 1, 0))                 # (H, W, I, O)
        needle_pe.proj.weight.data = Tensor(w_hwio.astype("float32"))

        if torch_pe.proj.bias is not None and needle_pe.proj.bias is not None:
            b_torch = torch_pe.proj.bias.detach().cpu().numpy()      # (O,)
            needle_pe.proj.bias.data = Tensor(b_torch.astype("float32"))

        # Copy norm weights if present
        if use_norm:
            gamma_torch = torch_pe.norm.weight.detach().cpu().numpy()  # (D,)
            beta_torch = torch_pe.norm.bias.detach().cpu().numpy()     # (D,)
            needle_pe.norm.weight.data = Tensor(gamma_torch.astype("float32"))
            needle_pe.norm.bias.data = Tensor(beta_torch.astype("float32"))

        return torch_pe, needle_pe

    # ---------- Single test case runner ----------

    def run_forward_case(
        B, C, H, W,
        img_size, patch_size,
        embed_dim,
        use_norm,
        flatten_embedding,
        atol=1e-5,
        rtol=1e-5,
    ):
        print(
            f"Case: B={B}, C={C}, H={H}, W={W}, "
            f"img_size={img_size}, patch_size={patch_size}, "
            f"embed_dim={embed_dim}, use_norm={use_norm}, "
            f"flatten={flatten_embedding}"
        )

        torch_pe, needle_pe = make_aligned_patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=C,
            embed_dim=embed_dim,
            use_norm=use_norm,
            flatten_embedding=flatten_embedding,
        )

        # Random input
        np.random.seed(0)
        x_np = np.random.randn(B, C, H, W).astype("float32")
        x_torch = torch.tensor(x_np)
        x_needle = Tensor(x_np)

        # Forward
        y_torch = torch_pe(x_torch).detach().cpu().numpy()
        y_needle = needle_pe(x_needle).numpy()

        # Compare
        assert y_torch.shape == y_needle.shape, (
            f"Shape mismatch: torch {y_torch.shape}, needle {y_needle.shape}"
        )
        max_diff = np.max(np.abs(y_torch - y_needle))
        print(f"  forward max diff: {max_diff:.6e}")
        assert np.allclose(y_torch, y_needle, atol=atol, rtol=rtol), "Forward mismatch!"
        print("  ✓ passed\n")

    # ---------- Run several cases ----------

    cases = [
        # B, C, H, W, img_size, patch_size, embed_dim, use_norm, flatten_embedding
        (2, 3, 32, 32, 32, 4, 16, False, True),
        (2, 3, 32, 32, 32, 4, 16, True, True),
        (1, 3, 64, 64, 64, 8, 32, True, True),
        (1, 3, 64, 64, 64, 16, 32, True, False),
    ]

    for case in cases:
        run_forward_case(*case)

    print("All PatchEmbed forward alignment tests passed ✅")