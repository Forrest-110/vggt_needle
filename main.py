import torch
from vggt_torch.models.vggt import VGGT as VGGT_torch
from needle import Tensor, nn, cuda
from vggt_needle.models.vggt import VGGT as VGGT_needle
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

# -----------------------------
# 1. Load pretrained SD from HF
# -----------------------------
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

model_torch = VGGT_torch()
vggt_sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu", model_dir='/data/hf_cache/')

model_torch.load_state_dict(vggt_sd)


# print("Loaded keys:", len(vggt_sd))
# for k in list(vggt_sd.keys())[:20]:
#     print("  ", k)
# print("...")


# -----------------------------
# 2. Initialize Needle model
# -----------------------------
model = VGGT_needle()
model_sd = model.state_dict()

# print("\nModel keys:", len(model_sd))
# for k in list(model_sd.keys())[:20]:
#     print("  ", k)
# print("...")


# # -----------------------------
# # 3. Match and load weights
# # -----------------------------
# def load_weights_needle(model: nn.Module, torch_sd: dict):
#     needle_sd = model.state_dict()

#     missing_keys = []
#     unexpected_keys = []
#     loaded_keys = []

#     # (A) Load matching keys (name + shape must match)
#     for name, torch_tensor in torch_sd.items():
#         if name not in needle_sd:
#             unexpected_keys.append(name)
#             continue

#         needle_tensor = needle_sd[name]

#         if tuple(torch_tensor.shape) != tuple(needle_tensor.shape):
#             if tuple(torch_tensor.unsqueeze(0).shape) != tuple(needle_tensor.shape):
#                 print(f"⚠️ Shape mismatch for {name}: "
#                     f"torch {tuple(torch_tensor.shape)} vs needle {tuple(needle_tensor.shape)}")
#                 continue

#         # Convert torch → numpy → needle
#         model.__dict__[name] = Tensor(
#             torch_tensor.detach().cpu().numpy(),
#             device=needle_tensor.device,
#             dtype=needle_tensor.dtype
#         ).broadcast_to(needle_tensor.shape)

#         loaded_keys.append(name)

#     # (B) Find missing model keys
#     for name in needle_sd.keys():
#         if name not in torch_sd:
#             missing_keys.append(name)

#     # # (C) Logging
#     # print("\n====== Weight Loading Report ======")
#     # print(f"Loaded keys:      {len(loaded_keys)}")
#     # print(f"Missing keys:     {len(missing_keys)}")
#     # print(f"Unexpected keys:  {len(unexpected_keys)}")

#     # if missing_keys:
#     #     print("\nMissing keys (not in checkpoint):")
#     #     for k in missing_keys[:20]:
#     #         print("  ", k)
#     #     if len(missing_keys) > 20:
#     #         print("  ...")

#     # if unexpected_keys:
#     #     print("\nUnexpected keys (not in model):")
#     #     for k in unexpected_keys[:20]:
#     #         print("  ", k)
#     #     if len(unexpected_keys) > 20:
#     #         print("  ...")

#     # print("===================================\n")

#     return loaded_keys, missing_keys, unexpected_keys


def sd_torch2needle(model: nn.Module, torch_sd: dict):
    needle_sd = model.state_dict()
    new_sd = {}
    # (A) Load matching keys (name + shape must match)
    for name, torch_tensor in torch_sd.items():
        if name not in needle_sd:
            raise ValueError(name)
        needle_tensor = needle_sd[name]

        if tuple(torch_tensor.shape) != tuple(needle_tensor.shape):
            if tuple(torch_tensor.unsqueeze(0).shape) != tuple(needle_tensor.shape):
                print(f"⚠️ Shape mismatch for {name}: "
                    f"torch {tuple(torch_tensor.shape)} vs needle {tuple(needle_tensor.shape)}")
                continue

        # Convert torch → numpy → needle
        new_sd[name] = Tensor(
            torch_tensor.detach().cpu().numpy(),
            device=needle_tensor.device,
            dtype=needle_tensor.dtype
        ).broadcast_to(needle_tensor.shape)
    return new_sd

# -----------------------------
# 4. Run the load
# -----------------------------
# print(model.aggregator.patch_embed.patch_embed.proj.weight.sum())
vggt_sd = sd_torch2needle(model, vggt_sd)
model.load_state_dict(vggt_sd, strict=False)
# print(f"Successfully loaded {len(loaded)} weights.")

# print(model.aggregator.patch_embed.patch_embed.proj.weight.sum())
# print(model_torch.aggregator.patch_embed.patch_embed.proj.weight.sum())
# exit()

device = cuda()
# -----------------------------
# 5. GPU forward inference demo
# -----------------------------
# NOTE: adjust shapes to whatever VGGT actually expects.
# Here I assume something like [B, S, C, H, W].
B, S, C, H, W = 1, 12, 3, 224, 224  # change if needed


# Create a random input on the same Needle device
dummy_np = np.random.randn(B, S, C, H, W).astype("float32")
dummy = Tensor(dummy_np, device=device)
dummy_torch = torch.from_numpy(dummy_np).cuda()

model = model.to(device)
model_torch.cuda()

# If your Needle nn has eval / no_grad equivalents, you can wrap them here.
print("Running forward pass on GPU (if enabled)...")
out_torch = model_torch(dummy_torch)
out = model(dummy)

# exit()
def print_dict(d, indent=0):
    """Recursively pretty-print a dict with shapes/types."""
    pad = "  " * indent

    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{pad}{k}:")
            print_dict(v, indent + 1)
    elif isinstance(d, (list, tuple)):
        print(f"{pad}{type(d).__name__} (len={len(d)}):")
        for i, item in enumerate(d):
            print(f"{pad}  [{i}]:")
            print_dict(item, indent + 2)
    else:
        # Print Tensor-like objects with .shape
        if hasattr(d, "shape"):
            try:
                print(f"{pad}{type(d)}, shape={tuple(d.shape)}")
            except Exception:
                print(f"{pad}{type(d)} (shape unreadable)")
        else:
            print(f"{pad}{type(d)}: {d}")

print_dict(out)
print()
print()
print()
print_dict(out_torch)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    raise TypeError(f"Cannot convert type {type(x)} to numpy")

def compare_struct(a, b, path="", atol=1e-5, rtol=1e-5):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"Key mismatch at {path}: {a.keys()} vs {b.keys()}"
        for k in a.keys():
            compare_struct(a[k], b[k], path + f".{k}", atol, rtol)

    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"Len mismatch at {path}: {len(a)} vs {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            compare_struct(ai, bi, path + f"[{i}]", atol, rtol)

    else:
        # Tensor-like leaf
        if hasattr(a, "shape") and hasattr(b, "shape"):
            assert tuple(a.shape) == tuple(b.shape), \
                f"Shape mismatch at {path}: {tuple(a.shape)} vs {tuple(b.shape)}"
            na, nb = to_numpy(a), to_numpy(b)
            if not np.allclose(na, nb, atol=atol, rtol=rtol):
                max_diff = float(np.max(np.abs(na - nb)))
                print(f"[WARN] Value mismatch at {path}, max |Δ| = {max_diff}")
        else:
            # Fallback for non-tensor leaves
            if a != b:
                print(f"[WARN] Non-tensor mismatch at {path}: {a} vs {b}")

compare_struct(out, out_torch)