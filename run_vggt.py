import time
import numpy as np
import torch

from needle import Tensor, nn, cuda
from vggt_needle.models.vggt import VGGT as VGGT_needle

# ---------------------------------------------------------------------
# 0. Repro & config
# ---------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)

# VGGT-1B default image shape
B, S, C, H, W = 1, 2, 3, 224, 224  # batch, sequence, channels, height, width

# ---------------------------------------------------------------------
# 2. Initialize Needle model & copy weights
# ---------------------------------------------------------------------
print("==> Initializing Needle VGGT...")
model_needle = VGGT_needle()
# ---------------------------------------------------------------------
# 3. Move models & data to GPU
# ---------------------------------------------------------------------
device_needle = cuda()

print("==> Moving models to GPU...")
model_needle = model_needle.to(device_needle)

# Create one shared random input on CPU → copy to both backends
print("==> Creating dummy input...")
dummy_np = np.random.randn(B, S, C, H, W).astype("float32")
dummy_needle = Tensor(dummy_np, device=device_needle)

for _ in range(3):
    _ = model_needle(dummy_needle)

from needle.autograd import print_op_stats, clean_op_stats
from needle.nn import reset_module_profile, print_module_profile
clean_op_stats()
reset_module_profile()

_ = model_needle(dummy_needle)
print_op_stats(top_k=30)
print_module_profile(top_k=30)


# from needle.autograd import print_noncompact_stats
# _ = model_needle(dummy_needle)
# print_noncompact_stats(top_k=30)
