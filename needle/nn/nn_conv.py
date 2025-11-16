"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride


        self.padding = padding
        self.weight = Parameter(
            init.kaiming_uniform(
                0, 0,
                shape=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                device=device, dtype=dtype
            )
        )
        if bias:
            bound = 1.0 / (self.in_channels * (self.kernel_size ** 2)) ** 0.5
            self.bias = Parameter(
                init.rand(self.out_channels, low=-bound, high=bound, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        x_nhwc = ops.transpose(ops.transpose(x, (1,2)), (2, 3))
        weight = self.weight.permute((2, 3, 1, 0)) + 0.0
        y_nhwc = ops.conv(x_nhwc, weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            b = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            y_nhwc = y_nhwc + b.broadcast_to(y_nhwc.shape)
        y_nchw = ops.transpose(ops.transpose(y_nhwc, (3, 2)), (2, 1))
        return y_nchw
    


class ConvTranspose2d(Module):
    """
    Multi-channel 2D transposed convolutional layer.

    - Expects NCHW input, returns NCHW output.
    - Wraps Needle's NHWC-based conv_transpose op.
    - No groups or dilation.
    - Only supports square kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: str = "float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(
            init.kaiming_uniform(
                0,
                0,
                shape=(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, ),
                dtype=dtype,
            )
        )

        if bias:
            # Match Conv's bias init: 1 / sqrt(in_channels * k^2)
            bound = 1.0 / (self.in_channels * (self.kernel_size ** 2)) ** 0.5
            self.bias = Parameter(
                init.rand(self.out_channels, low=-bound, high=bound, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: NCHW -> NHWC
        x_nhwc = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))  # (N, H, W, C_in)
        weight = self.weight.permute((2, 3, 1, 0)) + 0.0
        # Transposed conv in NHWC
        y_nhwc = ops.conv_transpose(
            x_nhwc,
            weight,
            stride=self.stride,
            padding=self.padding,
        )

        # Add bias if present: (1, 1, 1, C_out) broadcast to NHWC
        if self.bias is not None:
            b = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            y_nhwc = y_nhwc + b.broadcast_to(y_nhwc.shape)

        # NHWC -> NCHW
        y_nchw = ops.transpose(ops.transpose(y_nhwc, (3, 2)), (2, 1))
        return y_nchw