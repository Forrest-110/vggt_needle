#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



import argparse
import numpy as np

from needle import nn, init, Tensor
from needle import ops

from vggt.models.aggregator import Aggregator


def build_small_aggregator(
    img_size: int,
    patch_size: int,
    depth: int,
    embed_dim: int,
    num_heads: int,
    num_register_tokens: int,
    aa_block_size: int = 1,
    aa_order=None,
    rope_freq: int = 100,
) -> Aggregator:
    if aa_order is None:
        aa_order = ["frame", "global"]

    agg = Aggregator(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        block_fn=Aggregator.__mro__[0].__dict__.get("frame_blocks", None) or None,  # ignored, overridden below
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="conv",    # use conv PatchEmbed (lightweight)
        aa_order=aa_order,
        aa_block_size=aa_block_size,
        qk_norm=True,
        rope_freq=rope_freq,
        init_values=0.01,
    )
    return agg


def make_dummy_video(B: int, S: int, H: int, W: int) -> Tensor:
    """
    Create dummy video input: [B, S, 3, H, W], values in [0, 1].
    """
    x_np = np.random.rand(B, S, 3, H, W).astype("float32")
    return Tensor(x_np)


def test_aggregator_once(
    img_size: int = 112,
    patch_size: int = 16,
    depth: int = 4,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_register_tokens: int = 2,
    aa_block_size: int = 1,
    aa_order=None,
    B: int = 2,
    S: int = 3,
):
    print(
        f"\n=== Testing Aggregator: "
        f"img={img_size}, patch={patch_size}, depth={depth}, "
        f"embed_dim={embed_dim}, heads={num_heads}, regs={num_register_tokens}, "
        f"aa_block_size={aa_block_size}, B={B}, S={S} ==="
    )
    

    agg = Aggregator(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_register_tokens=num_register_tokens,
        patch_embed="dinov2_vits14_reg"
    )
    

    # dummy input
    x = make_dummy_video(B, S, img_size, img_size)
    print("Input shape:", x.shape)
    
    output_list, patch_start_idx = agg(x)
    print(f"patch_start_idx: {patch_start_idx}")

    # --- basic checks ---
    assert patch_start_idx == 1 + num_register_tokens, (
        f"patch_start_idx should be 1 + num_register_tokens ({1 + num_register_tokens}), "
        f"got {patch_start_idx}"
    )

    # depth vs output_list length:
    # aa_block_num = depth // aa_block_size
    # each aa_block contributes aa_block_size outputs
    # -> total outputs = depth
    assert len(output_list) == depth, f"Expected {depth} outputs, got {len(output_list)}"

    print(f"Number of outputs from aggregator: {len(output_list)}")

    # Inspect shapes
    first = output_list[0]
    print("First output shape:", first.shape)
    B_out, S_out, P_out, C_out = first.shape

    assert B_out == B, f"Batch size mismatch: {B_out} vs {B}"
    assert S_out == S, f"Sequence length mismatch: {S_out} vs {S}"

    # Compute expected number of patches per frame
    H_p = img_size // patch_size
    W_p = img_size // patch_size
    expected_P = H_p * W_p + patch_start_idx  # includes special tokens (camera + registers)
    assert P_out == expected_P, f"Expected P={expected_P}, got P={P_out}"

    # C_out should be 2 * embed_dim (concat frame & global)
    assert C_out == 2 * embed_dim, f"Expected channel dim={2 * embed_dim}, got {C_out}"

    print("Aggregator test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-register-tokens", type=int, default=2)
    parser.add_argument("--aa-block-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument(
        "--aa-order",
        type=str,
        nargs="+",
        default=["frame", "global"],
        help="Alternating attention order, e.g., --aa-order frame global frame",
    )
    args = parser.parse_args()

    test_aggregator_once(
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_register_tokens=args.num_register_tokens,
        aa_block_size=args.aa_block_size,
        aa_order=args.aa_order,
        B=args.batch_size,
        S=args.seq_len,
    )

    print("\nAll Aggregator tests finished successfully.")


if __name__ == "__main__":
    main()
