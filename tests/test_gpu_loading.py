#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np

# Adjust these imports to your project layout
from needle import backend_ndarray as nd
from needle import Tensor
from needle.nn import Linear  # or wherever your Module base class lives


def test_tensor_gpu():
    print("=== Tensor GPU test ===")
    cpu_dev = nd.cpu_numpy()
    cuda_dev = nd.cuda()

    print(f"CPU device: {cpu_dev}")
    print(f"CUDA device: {cuda_dev}")

    if not cuda_dev.enabled():
        raise NotImplementedError

    x_cpu = Tensor(np.random.randn(2, 3).astype("float32"), device=cpu_dev)
    print("x_cpu.device:", x_cpu.device)

    x_gpu = x_cpu.to(cuda_dev)
    print("x_gpu.device:", x_gpu.device)

    assert x_cpu.device == cpu_dev, "x_cpu should be on CPU device"
    assert x_gpu.device == cuda_dev, "x_gpu should be on CUDA device"

    # Ensure original is unchanged
    assert x_cpu.device != x_gpu.device, "CPU and GPU tensors must be on different devices"
    print("Tensor GPU move: OK\n")


def test_module_gpu():
    print("=== Module GPU test ===")
    cpu_dev = nd.cpu_numpy()
    cuda_dev = nd.cuda()

    if not cuda_dev.enabled():
        print("CUDA backend not available; skipping module GPU test.")
        return

    # Create module on CPU
    model = Linear(3, 4, device=cpu_dev)
    print("Initial model.param devices:")
    print("  weight.device:", model.weight.device)
    print("  bias.device:", model.bias.device)

    assert model.weight.device == cpu_dev
    assert model.bias.device == cpu_dev

    # Move model to GPU
    model_gpu = model.to(cuda_dev)
    print("After model.to(cuda) devices:")
    print("  weight.device:", model_gpu.weight.device)
    print("  bias.device:", model_gpu.bias.device)

    assert model_gpu.weight.device == cuda_dev, "weight should be on CUDA after model.to(cuda)"
    assert model_gpu.bias.device == cuda_dev, "bias should be on CUDA after model.to(cuda)"

    # Run a forward pass on GPU to check it actually works
    x_gpu = Tensor(np.random.randn(5, 3).astype("float32"), device=cuda_dev)
    y_gpu = model_gpu(x_gpu)
    print("Forward on GPU OK, output shape:", y_gpu.shape)

    print("Module GPU move: OK\n")


if __name__ == "__main__":
    print("Running GPU loading tests...\n")
    # test_tensor_gpu()
    test_module_gpu()
    print("All done.")
