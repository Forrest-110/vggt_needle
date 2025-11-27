import time
import numpy as np
import torch

from vggt_torch.models.vggt import VGGT as VGGT_torch
from needle import Tensor, nn, cuda
from vggt_needle.models.vggt import VGGT as VGGT_needle

# ---------------------------------------------------------------------
# 0. Repro & config
# ---------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)

WARMUP_ITERS = 3
BENCH_ITERS = 3

# VGGT-1B default image shape
B, S, C, H, W = 1, 2, 3, 224, 224  # batch, sequence, channels, height, width

_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_HF_CACHE_DIR = "/data/hf_cache/"


# ---------------------------------------------------------------------
# 1. Load pretrained PyTorch model
# ---------------------------------------------------------------------
print("==> Loading PyTorch VGGT and HF state dict...")
model_torch = VGGT_torch()
vggt_sd_torch = torch.hub.load_state_dict_from_url(
    _URL, map_location="cpu", model_dir=_HF_CACHE_DIR
)
model_torch.load_state_dict(vggt_sd_torch)
print("   PyTorch state dict loaded.")


# ---------------------------------------------------------------------
# 2. Initialize Needle model & copy weights
# ---------------------------------------------------------------------
print("==> Initializing Needle VGGT...")
model_needle = VGGT_needle()

def sd_torch2needle(model: nn.Module, torch_sd: dict):
    needle_sd = model.state_dict()
    new_sd = {}

    for name, torch_tensor in torch_sd.items():
        if name not in needle_sd:
            raise ValueError(f"Key {name} not found in Needle state dict")

        needle_tensor = needle_sd[name]

        if tuple(torch_tensor.shape) != tuple(needle_tensor.shape):
            # Allow for an extra leading dim (e.g. [C,H,W] vs [1,C,H,W])
            if tuple(torch_tensor.unsqueeze(0).shape) != tuple(needle_tensor.shape):
                print(
                    f"⚠️ Shape mismatch for {name}: "
                    f"torch {tuple(torch_tensor.shape)} vs needle {tuple(needle_tensor.shape)}"
                )
                continue
            torch_tensor = torch_tensor.unsqueeze(0)

        new_sd[name] = Tensor(
            torch_tensor.detach().cpu().numpy(),
            device=needle_tensor.device,
            dtype=needle_tensor.dtype,
        ).broadcast_to(needle_tensor.shape)

    return new_sd

print("==> Converting PyTorch state dict → Needle...")
vggt_sd_needle = sd_torch2needle(model_needle, vggt_sd_torch)
model_needle.load_state_dict(vggt_sd_needle, strict=False)
print("   Needle state dict loaded.")


# ---------------------------------------------------------------------
# 3. Move models & data to GPU
# ---------------------------------------------------------------------
device_needle = cuda()

print("==> Moving models to GPU...")
model_torch = model_torch.cuda()
model_needle = model_needle.to(device_needle)

model_torch.eval()
# If your Needle nn has eval(), call it; otherwise it's a no-op:
if hasattr(model_needle, "eval"):
    model_needle.eval()

# Enable CuDNN autotuner for PyTorch convs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Create one shared random input on CPU → copy to both backends
print("==> Creating dummy input...")
dummy_np = np.random.randn(B, S, C, H, W).astype("float32")

dummy_torch = torch.from_numpy(dummy_np).cuda(non_blocking=True)
dummy_needle = Tensor(dummy_np, device=device_needle)

def needle_sync():
    """Best-effort device sync for Needle CUDA backend."""
    try:
        # Some Needle backends expose synchronize() on the device module
        device_needle.synchronize()
    except Exception:
        # If not available, we just skip explicit sync
        pass


# ---------------------------------------------------------------------
# 4. Benchmark helpers
# ---------------------------------------------------------------------
def benchmark_torch(model, x, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    total = t1 - t0
    avg = total / iters
    return avg, total

from tqdm import tqdm
def benchmark_needle(model, x, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    # Warmup
    for _ in tqdm(range(warmup)):
        _ = model(x)
    # needle_sync()

    t0 = time.perf_counter()
    for _ in tqdm(range(iters)):
        _ = model(x)
    # needle_sync()
    t1 = time.perf_counter()

    total = t1 - t0
    avg = total / iters
    return avg, total


# ---------------------------------------------------------------------
# 5. Run benchmarks
# ---------------------------------------------------------------------
print("==> Benchmarking PyTorch VGGT...")
avg_torch, total_torch = benchmark_torch(model_torch, dummy_torch)
print("==> Benchmarking Needle VGGT...")
avg_needle, total_needle = benchmark_needle(model_needle, dummy_needle)

# ---------------------------------------------------------------------
# 6. Report results
# ---------------------------------------------------------------------
def pretty_ms(sec):
    return sec * 1000.0

num_tokens = B * S  # rough notion of "frames" or tokens per forward

print("\n================= Benchmark Results =================")
print(f"Config: B={B}, S={S}, C={C}, H={H}, W={W}")
print(f"Warmup iters : {WARMUP_ITERS}")
print(f"Measure iters: {BENCH_ITERS}\n")

print("PyTorch VGGT:")
print(f"  Avg latency : {pretty_ms(avg_torch):.3f} ms / forward")
print(f"  Throughput  : {num_tokens / avg_torch:.2f} tokens-per-second (approx)")

print("\nNeedle VGGT:")
print(f"  Avg latency : {pretty_ms(avg_needle):.3f} ms / forward")
print(f"  Throughput  : {num_tokens / avg_needle:.2f} tokens-per-second (approx)")

speedup = avg_torch / avg_needle if avg_needle > 0 else float("inf")
print("\n-----------------------------------------------------")
print(f"Needle vs PyTorch speed ratio (latency): {speedup:.3f}x ( >1.0 means Needle faster )")
print("=====================================================\n")
