import torch
from needle import Tensor, nn
from vggt.models.vggt import VGGT

# -----------------------------
# 1. Load pretrained SD from HF
# -----------------------------
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
vggt_sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")

print("Loaded keys:", len(vggt_sd))
for k in list(vggt_sd.keys())[:20]:
    print("  ", k)
print("...")


# -----------------------------
# 2. Initialize Needle model
# -----------------------------
model = VGGT()
model_sd = model.state_dict()

print("\nModel keys:", len(model_sd))
for k in list(model_sd.keys())[:20]:
    print("  ", k)
print("...")


# -----------------------------
# 3. Match and load weights
# -----------------------------
def load_weights_needle(model: nn.Module, torch_sd: dict):
    needle_sd = model.state_dict()

    missing_keys = []
    unexpected_keys = []
    loaded_keys = []

    # (A) Load matching keys (name + shape must match)
    for name, torch_tensor in torch_sd.items():
        if name not in needle_sd:
            unexpected_keys.append(name)
            continue

        needle_tensor = needle_sd[name]

        if tuple(torch_tensor.shape) != tuple(needle_tensor.shape):
            if tuple(torch_tensor.unsqueeze(0).shape) != tuple(needle_tensor.shape):
                print(f"⚠️ Shape mismatch for {name}: "
                    f"torch {tuple(torch_tensor.shape)} vs needle {tuple(needle_tensor.shape)}")
                continue

        # Convert torch → numpy → needle
        model.__dict__[name] = Tensor(
            torch_tensor.detach().cpu().numpy(),
            device=needle_tensor.device,
            dtype=needle_tensor.dtype
        ).broadcast_to(needle_tensor.shape)

        loaded_keys.append(name)

    # (B) Find missing model keys
    for name in needle_sd.keys():
        if name not in torch_sd:
            missing_keys.append(name)

    # (C) Logging
    print("\n====== Weight Loading Report ======")
    print(f"Loaded keys:      {len(loaded_keys)}")
    print(f"Missing keys:     {len(missing_keys)}")
    print(f"Unexpected keys:  {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing keys (not in checkpoint):")
        for k in missing_keys[:20]:
            print("  ", k)
        if len(missing_keys) > 20:
            print("  ...")

    if unexpected_keys:
        print("\nUnexpected keys (not in model):")
        for k in unexpected_keys[:20]:
            print("  ", k)
        if len(unexpected_keys) > 20:
            print("  ...")

    print("===================================\n")

    return loaded_keys, missing_keys, unexpected_keys


# -----------------------------
# 4. Run the load
# -----------------------------
loaded, missing, unexpected = load_weights_needle(model, vggt_sd)
print(f"Successfully loaded {len(loaded)} weights.")
