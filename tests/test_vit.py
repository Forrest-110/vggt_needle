#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

"""
Simple test script for DinoVisionTransformer / ViT in Needle.

Usage:
    python test_vit.py --model vit_small --img-size 224 --batch-size 2
"""

import argparse
import math

from needle import init, ops, Tensor
from needle import nn

from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2, DinoVisionTransformer


def build_model(name: str, img_size: int, chunked: bool = False) -> nn.Module:
    if name == "vit_small":
        fn = vit_small
    elif name == "vit_base":
        fn = vit_base
    elif name == "vit_large":
        fn = vit_large
    elif name == "vit_giant2":
        fn = vit_giant2
    else:
        raise ValueError(f"Unknown model name: {name}")

    if chunked:
        # keep depth from preset, just enable block_chunks=2 as a basic test
        model = fn(img_size=img_size, block_chunks=2, num_register_tokens=4, interpolate_antialias=True,
                interpolate_offset=0.0,
                init_values=1.0)
    else:
        model = fn(img_size=img_size, block_chunks=0, num_register_tokens=4,
                   interpolate_antialias=True,
                interpolate_offset=0.0,
                init_values=1.0)
    return model


def make_dummy_input(batch_size: int, img_size: int, in_chans: int = 3) -> Tensor:
    # (B, C, H, W)
    return init.randn(batch_size, in_chans, img_size, img_size)


def make_dummy_masks(model: DinoVisionTransformer, batch_size: int, img_size: int) -> Tensor:
    """
    Build random 0/1 masks with shape (B, num_patches).
    """
    num_patches = model.patch_embed.num_patches
    # random in [0,1), then threshold
    m = init.rand(batch_size, num_patches)
    masks = ops.relu(ops.sign(m - 0.5))
    return masks


def test_forward_backward(model_name: str, img_size: int, batch_size: int, chunked: bool = False):
    print(f"\n=== Testing {model_name} (img_size={img_size}, batch_size={batch_size}, chunked={chunked}) ===")
    model = build_model(model_name, img_size=img_size, chunked=chunked)

    x = make_dummy_input(batch_size, img_size)
    print("Input shape:", x.shape)

    # ----- Forward without masks -----
    out = model(x)  # head(x_norm_clstoken)
    print("Output shape (no masks):", out.shape)

    assert out.shape[0] == batch_size, "Batch size mismatch in output"
    # cls token dim should match embed_dim / num_features
    assert out.shape[1] == model.embed_dim, f"Output dim must be embed_dim, {out.shape}, {model.embed_dim}"



    masks = make_dummy_masks(model, batch_size, img_size)
    out_masked = model(x, masks=masks)
    print("Output shape (with masks):", out_masked.shape)
    assert out_masked.shape == out.shape

    # ----- Test forward_features -----
    feats = model.forward_features(x)
    print("forward_features keys:", list(feats.keys()))
    for k, v in feats.items():
        if isinstance(v, Tensor):
            print(f"  {k}: {v.shape}")

    # Basic sanity checks on feature dict
    cls_tok = feats["x_norm_clstoken"]
    patch_tok = feats["x_norm_patchtokens"]
    assert cls_tok.shape[0] == batch_size
    assert patch_tok.shape[0] == batch_size

    # number of patch tokens should match num_patches
    num_patches = model.patch_embed.num_patches
    assert patch_tok.shape[1] == num_patches, f"Expected {num_patches} patch tokens, got {patch_tok.shape[1]}"

    # ----- Test get_intermediate_layers -----
    print("Testing get_intermediate_layers...")
    outs_last2 = model.get_intermediate_layers(x, n=2, reshape=False, return_class_token=True)
    print(f"  get_intermediate_layers n=2, reshape=False -> {len(outs_last2)} elements")
    for i, (patches, cls) in enumerate(outs_last2):
        print(f"    layer[{i}] patches: {patches.shape}, cls: {cls.shape}")

    # reshape=True path
    outs_reshaped = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
    print(f"  get_intermediate_layers n=1, reshape=True -> {len(outs_reshaped)} elements")
    B = batch_size
    for i, y in enumerate(outs_reshaped):
        print(f"    reshaped[{i}] shape: {y.shape}")
        # expect (B, C, H', W') with H'*W' == num_patches
        assert y.shape[0] == B
        h_p = img_size // model.patch_size
        w_p = img_size // model.patch_size
        assert y.shape[2] == h_p and y.shape[3] == w_p, "Reshaped spatial dims mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large", "vit_giant2"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-chunked", action="store_true",
                        help="Also test a chunked-block configuration.")
    args = parser.parse_args()

    # non-chunked path
    test_forward_backward(args.model, args.img_size, args.batch_size, chunked=False)

    # # optional: chunked blocks path
    # if args.test_chunked:
    #     test_forward_backward(args.model, args.img_size, args.batch_size, chunked=True)

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()