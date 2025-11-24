import torch

def print_cuda_mem(tag: str):
    """Print global CUDA memory usage using torch, regardless of backend."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    gb = 1024 ** 3
    print(
        f"[MEM] {tag}: "
        f"used={used / gb:.2f} GB, free={free / gb:.2f} GB, total={total / gb:.2f} GB"
    )