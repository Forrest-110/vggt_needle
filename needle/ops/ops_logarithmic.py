from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        z_max = array_api.max(Z, axis=-1, keepdims=True)
        shifted = Z - z_max
        lse = array_api.log(array_api.sum(array_api.exp(shifted), axis=-1, keepdims=True)) + z_max
        return Z - lse

    def gradient(self, out_grad: Tensor, node: Tensor):
        (Z,) = node.inputs
        in_shape = Z.shape

        softmax = exp(node)
        sum_g = summation(out_grad, axes=(-1,))
        reshaped = tuple(list(in_shape[:-1]) + [1])
        sum_g_b = broadcast_to(reshape(sum_g, reshaped), in_shape)

        return add(out_grad, negate(multiply(softmax, sum_g_b)))


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        axes = self.axes
        if axes is None:
            axes = tuple(range(Z.ndim))
        elif not isinstance(axes, (tuple, list)):
            axes = (axes,)

        z_max = Z.max(axis=axes, keepdims=True)
        shifted = Z - array_api.broadcast_to(z_max, Z.shape)
        lse_keep = array_api.log(array_api.sum(array_api.exp(shifted), axis=axes, keepdims=True)) + z_max
        
        lse = array_api.squeeze(lse_keep, axis=axes)
        # if lse.shape == ():                         
        #     lse = float(lse.numpy())
        return lse

    def gradient(self, out_grad: Tensor, node: Tensor):
        out_grad = out_grad + 0.0
        (Z,) = node.inputs
        in_shape = Z.shape
        ndim = len(in_shape)

        axes = self.axes
        if axes is None:
            reduce_axes = tuple(range(ndim))
        else:
            if not isinstance(axes, (tuple, list)):
                axes = (axes,)
            reduce_axes = tuple(sorted(ax if ax >= 0 else ax + ndim for ax in axes))

        reshaped = tuple(1 if i in reduce_axes else in_shape[i] for i in range(ndim))


        lse = LogSumExp(self.axes)(Z)
        lse_b = broadcast_to(reshape(lse, reshaped), in_shape)
        softmax_term = exp(add(Z, negate(lse_b)))
        og = broadcast_to(reshape(out_grad, reshaped), in_shape)

        return multiply(og, softmax_term)


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)


class Softmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # Numerically stable softmax over the last dimension
        z_max = Z.max(axis=-1, keepdims=True)
        shifted = Z - z_max.broadcast_to(Z.shape)
        exp_shifted = array_api.exp(shifted)
        denom = array_api.sum(exp_shifted, axis=-1, keepdims=True).broadcast_to(exp_shifted.shape)
        return exp_shifted / denom

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Let y = softmax(Z) (i.e., node), and g = out_grad.
        The gradient w.r.t Z is:

            dL/dZ = y * (g - sum_j g_j * y_j)

        where the sum is over the last dimension.
        """
        (Z,) = node.inputs
        in_shape = Z.shape

        y = node  # softmax(Z) output

        # sum_j g_j * y_j along last dimension
        gy = multiply(out_grad, y)
        sum_gy = summation(gy, axes=(-1,))

        # reshape/broadcast to match input shape
        reshaped = tuple(list(in_shape[:-1]) + [1])
        sum_gy_b = broadcast_to(reshape(sum_gy, reshaped), in_shape)

        # g - sum_j g_j * y_j
        diff = add(out_grad, negate(sum_gy_b))

        # y * (g - sum_j g_j * y_j)
        return multiply(y, diff)


def softmax(a: Tensor) -> Tensor:
    return Softmax()(a)

class Sigmoid(TensorOp):
    def compute(self, x):
        # numerically stable sigmoid: 1 / (1 + exp(-x))
        return (1 + array_api.exp(-x)) ** -1

    def gradient(self, out_grad, node):
        # derivative: sigmoid(x) * (1 - sigmoid(x))
        y = sigmoid(node.inputs[0])    # y = sigmoid(x)
        return out_grad * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)