"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy
import math
# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar
    def gradient(self, out_grad, node):
        (a,) = node.inputs
        grad_a = out_grad * self.scalar * (a ** (self.scalar - 1))
        return (grad_a,)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = out_grad * (-a / (b ** 2))
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        grad_a = out_grad / self.scalar
        return (grad_a,)

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            i, j = self.axes
            return array_api.swapaxes(a, i, j)

    def gradient(self, out_grad, node):
        if self.axes is None:
            return Transpose()(out_grad)
        else:
            i, j = self.axes
            return Transpose((i, j))(out_grad)


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Permute(TensorOp):
    """
    Permute dimensions using repeated transpose operations.
    Equivalent to PyTorch's tensor.permute(dims).

    Example:
        x: (B, H, N, D)
        permute dims=(0, 2, 1, 3)
        -> (B, N, H, D)
    """
    def __init__(self, dims: tuple):
        assert isinstance(dims, (list, tuple)), "dims must be a tuple or list"
        self.dims = tuple(dims)

    def compute(self, a):
        # apply pairwise transposes to reach target permutation
        out = a
        current_order = list(range(out.ndim))

        for i, target_axis in enumerate(self.dims):
            j = current_order.index(target_axis)
            if i != j:
                # swap axis i and j
                out = array_api.swapaxes(out, i, j)
                current_order[i], current_order[j] = current_order[j], current_order[i]

        return out

    def gradient(self, out_grad, node):
        # inverse permutation: dims^{-1}
        dims = self.dims
        inv = [0] * len(dims)
        for i, d in enumerate(dims):
            inv[d] = i

        return Permute(tuple(inv))(out_grad)


def permute(a, dims):
    return Permute(dims)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        out_grad = out_grad + 0.0
        (a,) = node.inputs
        return (reshape(out_grad, a.shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        in_shape = a.shape
        out_shape = out_grad.shape
        pad_len = len(out_shape) - len(in_shape)
        padded_in_shape = (1,) * pad_len + in_shape
        reduce_axes = []
        for i, (in_dim, out_dim) in enumerate(zip(padded_in_shape, out_shape)):
            if in_dim == 1 and out_dim > 1:
                reduce_axes.append(i)
            elif in_dim == 0 and out_dim > 0:
                reduce_axes.append(i)
        if len(reduce_axes) == 0:
            grad = out_grad
        else: 
            grad = summation(out_grad, axes=tuple(reduce_axes))
        grad = reshape(grad, in_shape)
        return (grad,)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        in_shape = a.shape
        ndim = len(in_shape)
        axes = self.axes
        if axes is None:
            reduce_axes = tuple(range(ndim))
        else:
            if not isinstance(axes, (tuple, list)):
                axes = (axes,)
            reduce_axes = tuple(sorted([ax if ax >= 0 else ax + ndim for ax in axes]))

        reshaped = []
        for i in range(ndim):
            if i in reduce_axes:
                reshaped.append(1)
            else:
                reshaped.append(in_shape[i])
        reshaped = tuple(reshaped)

        out = reshape(out_grad, reshaped)
        out = broadcast_to(out, in_shape)
        return (out,)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        
        if len(a.shape) == 2:
            a = a.reshape((1, a.shape[0], a.shape[1]))
        if len(b.shape) == 2:
            b = b.reshape((1, b.shape[0], b.shape[1]))

        a_shape = a.shape
        b_shape = b.shape

        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        
        if len(a_batch) < len(b_batch):
            pad = (1,) * (len(b_batch) - len(a_batch))
            a = a.reshape(pad + a_shape)
            a_shape = a.shape
            a_batch = a_shape[:-2]
        elif len(b_batch) < len(a_batch):
            pad = (1,) * (len(a_batch) - len(b_batch))
            b = b.reshape(pad + b_shape)
            b_shape = b.shape
            b_batch = b_shape[:-2]
        
        out_batch = []
        for da, db in zip(a_batch, b_batch):
            if da == 1:
                out_batch.append(db)
            elif db == 1:
                out_batch.append(da)
            elif da == db:
                out_batch.append(da)
            else:
                raise ValueError(
                    f"matmul batch dims not broadcastable: {a_shape} vs {b_shape}"
                )
        out_batch = tuple(out_batch)

        if a.shape[:-2] != out_batch:
            a = a.broadcast_to(out_batch + a.shape[-2:]).compact()
        if b.shape[:-2] != out_batch:
            b = b.broadcast_to(out_batch + b.shape[-2:]).compact()

        *batch, m, k = a.shape
        *_, k2, n = b.shape
        assert k == k2, f"Incompatible matmul shapes: {a.shape} @ {b.shape}"

        B = 1
        for d in batch:
            B *= d

        a2 = a.reshape((B, m, k))
        
        b2 = b.reshape((B, k, n))

        out2 = array_api.empty((B, m, n))
        for i in range(B):
            out2[i, :, :] = (a2[i, :, :].compact().reshape((m, k)) @ b2[i, :, :].compact().reshape((k, n))).reshape((1, m, n))   # (m, n)

        out = out2.reshape(tuple(batch) + (m, n))
        return out

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = matmul(out_grad, Transpose((-1, -2))(b))
        grad_b = matmul(Transpose((-1, -2))(a), out_grad)
        def sum_to_shape(t, target_shape):
            t_shape = t.shape
            lead = len(t_shape) - len(target_shape)
            if lead > 0:
                t = summation(t, axes=tuple(range(lead)))
                t_shape = t.shape
            reduce_axes = []
            for i, (td, sd) in enumerate(zip(t_shape, target_shape)):
                if sd == 1 and td != 1:
                    reduce_axes.append(i)
            if reduce_axes:
                t = summation(t, axes=tuple(reduce_axes))
            if t.shape != target_shape:
                t = reshape(t, target_shape)
            return t

        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        return (out_grad / a,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        return (out_grad * exp(a))


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        mask = a.realize_cached_data() > 0
        return (out_grad * mask,)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        tmp = tanh(a)
        return (out_grad - out_grad * tmp * tmp,)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ref = args[0]
        ndim = len(ref.shape)
        axis = self.axis if self.axis >= 0 else self.axis + (ndim + 1)
        out_shape = tuple(ref.shape[:axis]) + (len(args),) + tuple(ref.shape[axis:])
        out = array_api.empty(out_shape, device=ref.device)

        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis)
        for i, t in enumerate(args):
            idx = tuple(left + [i] + right)
            out[idx] = t
        return out

    def gradient(self, out_grad, node):
        grads = split(out_grad, self.axis)
        return make_tuple(*grads)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ndim = len(A.shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        n = A.shape[axis]
        parts = []
        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis - 1)
        part_shape = tuple(A.shape[:axis] + A.shape[axis + 1:])
        for i in range(n):
            idx = tuple(left + [slice(i, i + 1)] + right)
            piece = A[idx].compact().reshape(part_shape)
            parts.append(piece)
        return tuple(parts)

    def gradient(self, out_grad, node):
        return (stack(list(out_grad), self.axis),)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None or len(self.axes) == 0:
            return a.compact()
        return array_api.flip(a, self.axes)


    def gradient(self, out_grad, node):
        return (flip(out_grad, self.axes),)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0 or len(self.axes) == 0:
            return a.compact()

        shape = list(a.shape)
        ndim = len(shape)
        d = self.dilation
        new_shape = []
        for i in range(ndim):
            if i in self.axes:
                new_shape.append(shape[i] + shape[i] * d)
            else:
                new_shape.append(shape[i])
        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        slices = []
        for i in range(ndim):
            if i in self.axes:
                slices.append(slice(0, new_shape[i], d + 1))
            else:
                slices.append(slice(None))
        out[tuple(slices)] = a
        return out

    def gradient(self, out_grad, node):
        return (undilate(out_grad, self.axes, self.dilation),)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0 or len(self.axes) == 0:
            return a.compact()

        ndim = len(a.shape)
        d = self.dilation
        slices = []
        for i in range(ndim):
            if i in self.axes:
                slices.append(slice(0, a.shape[i], d + 1))
            else:
                slices.append(slice(None))
        return a[tuple(slices)].compact()

    def gradient(self, out_grad, node):
        return (dilate(out_grad, self.axes, self.dilation),)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

def _one_hot_tensor(n: int, idx: int, device) -> Tensor:
    # NDArray constant then wrap as Tensor (no NumPy)
    v_nd = array_api.full((n,), 0.0, device=device)
    v_nd[idx] = 1.0
    return Tensor(v_nd, device=device)

def _eye_tensor(n: int, device) -> Tensor:
    rows = [ _one_hot_tensor(n, i, device) for i in range(n) ] 
    return stack(rows, axis=0)  

def _zeros_tensor(shape: tuple[int, ...], device) -> Tensor:
    return Tensor(array_api.full(shape, 0.0, device=device), device=device)

def _build_selector_kernel_tensor(K: int, C_in: int, i: int, j: int, device) -> Tensor:
    I = _eye_tensor(C_in, device)                        
    Z = _zeros_tensor((C_in, C_in), device)              
    rows = []
    for r in range(K):
        cols = []
        for c in range(K):
            cols.append(I if (r == i and c == j) else Z)  
        rows.append(stack(cols, axis=0))                
    return stack(rows, axis=0)  

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        s = self.stride
        p = self.padding

        N, H, W, C_in = A.shape
        K, K2, C_in_w, C_out = B.shape

        if p > 0:
            A_pad = A.pad(((0, 0), (p, p), (p, p), (0, 0)))
        else:
            A_pad = A

        Hp, Wp = A_pad.shape[1], A_pad.shape[2]
        H_out = (Hp - K) // s + 1
        W_out = (Wp - K) // s + 1

        Y = array_api.full((N, H_out, W_out, C_out), 0.0, device=A.device)

        for i in range(K):
            for j in range(K):
                A_ij = A_pad[:, i : i + H_out * s : s, j : j + W_out * s : s, :]
                W_ij = B[i, j, :, :]
                A_e = A_ij.compact().reshape((N, H_out, W_out, C_in, 1)).broadcast_to((N, H_out, W_out, C_in, C_out))
                W_e = W_ij.compact().reshape((1, 1, 1, C_in, C_out)).broadcast_to((N, H_out, W_out, C_in, C_out))
                contrib = (A_e * W_e).sum(axis=3)

                Y = Y + contrib

        return Y.compact()


    def gradient(self, out_grad, node):

        out_grad = out_grad + 0.0
        X, W = node.inputs
        s, p = self.stride, self.padding
        K, _, C_in, C_out = W.shape

        G = dilate(out_grad, axes=(1, 2), dilation=s - 1) if s > 1 else out_grad
        W_flip = flip(W, axes=(0, 1))              
        W_flip_T = transpose(W_flip, (3, 2))   
        pad_back = K - 1 - p
        dX = conv(G, W_flip_T, stride=1, padding=pad_back)

        N, H, Wsp, _ = X.shape
        N2, Hout, Wout, Cout = out_grad.shape
        assert N == N2 and Cout == C_out
  
        dW_rows = []
        for i in range(K):
            row = []
            for j in range(K):
                E_ij = _build_selector_kernel_tensor(K, C_in, i, j, device=X.device)  # (K,K,Cin,Cin)
                T_ij = conv(X, E_ij, stride=s, padding=p)
                T_e = reshape(T_ij, (N, Hout, Wout, C_in, 1)).broadcast_to((N, Hout, Wout, C_in, C_out))
                G_e = reshape(out_grad, (N, Hout, Wout, 1, C_out)).broadcast_to((N, Hout, Wout, C_in, C_out))
                dW_ij = summation(T_e * G_e, axes=(0, 1, 2)) 
                row.append(dW_ij)
            dW_rows.append(stack(row, axis=0))       
        dW = stack(dW_rows, axis=0)               

        return dX, dW


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

class ConvTranspose(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        """
        A: (N, H_in, W_in, C_out)   -- acts like 'grad output' of Conv
        B: (K, K, C_in, C_out)      -- same layout as Conv weights
        Returns:
            Y: (N, H_out, W_out, C_in)
        where:
            H_out = (H_in - 1) * stride - 2 * padding + K
            W_out = (W_in - 1) * stride - 2 * padding + K
        """
        s = self.stride
        p = self.padding

        N, H_in, W_in, C_out = A.shape
        K, K2, C_in, C_out_w = B.shape
        assert K == K2 and C_out == C_out_w

        # Output spatial size (no output_padding)
        H_out = (H_in - 1) * s - 2 * p + K
        W_out = (W_in - 1) * s - 2 * p + K

        Y = array_api.full((N, H_out, W_out, C_in), 0.0, device=A.device)

        # Naive but clear implementation:
        # For each input position (ih, iw), spread its value to output
        # positions according to stride and kernel.
        for n in range(N):
            for ih in range(H_in):
                for iw in range(W_in):
                    a_vec = A[n, ih, iw, :].compact()              # (C_out,)
                    if a_vec is None:
                        continue
                    for kh in range(K):
                        oh = ih * s - p + kh
                        if oh < 0 or oh >= H_out:
                            continue
                        for kw in range(K):
                            ow = iw * s - p + kw
                            if ow < 0 or ow >= W_out:
                                continue
                            # B[kh, kw] : (C_in, C_out)
                            W_ij = B[kh, kw, :, :]        # (C_in, C_out)
                            # out_vec[c_in] = sum_c_out a_vec[c_out] * W_ij[c_in, c_out]
                            # -> (C_in,)
        
                            prod = (W_ij * a_vec.reshape((1, C_out)).broadcast_to(W_ij.shape)).sum(axis=-1)
                            Y[n, oh, ow, :] = Y[n, oh, ow, :] + prod.broadcast_to(Y[n, oh, ow, :].shape)

        return Y.compact()

    def gradient(self, out_grad, node):
        """
        Let forward be:  Z = ConvTranspose(stride=s, padding=p)(A, W)
        where:
          A: (N, H_in,  W_in,  C_out)
          W: (K, K, C_in, C_out)
          Z: (N, H_out, W_out, C_in)

        We use the adjoint relationship between Conv and ConvTranspose:
          <Conv(X, W), A> = <X, ConvTranspose(A, W)>

        => gradient wrt A uses Conv forward
        => gradient wrt W uses the same dW formula as Conv, but with
           X := out_grad (Z) and out_grad := A.
        """
        A, W = node.inputs
        s, p = self.stride, self.padding

        # ---------- dA: Conv with same W, stride, padding ----------
        # out_grad: (N, H_out, W_out, C_in)  (same as Z)
        # A:       (N, H_in,  W_in,  C_out)
        # Conv(out_grad, W) -> (N, H_in, W_in, C_out) == shape of A
        dA = conv(out_grad, W, stride=s, padding=p)

        # ---------- dW: reuse Conv's dW logic, with X = out_grad, out_grad = A ----------
        X = out_grad
        N, H, Wsp, _ = X.shape
        N2, Hout, Wout, Cout = A.shape
        K, _, C_in, C_out = W.shape
        assert N == N2 and Cout == C_out

        dW_rows = []
        for i in range(K):
            row = []
            for j in range(K):
                # Same selector kernel as in Conv.gradient
                E_ij = _build_selector_kernel_tensor(K, C_in, i, j, device=X.device)  # (K,K,C_in,C_in)
                # Conv over X with selector picks out shifted X features
                T_ij = conv(X, E_ij, stride=s, padding=p)  # (N, H_in, W_in, C_in) == (N,Hout,Wout,C_in)

                T_e = reshape(T_ij, (N, Hout, Wout, C_in, 1)).broadcast_to((N, Hout, Wout, C_in, C_out))
                G_e = reshape(A,   (N, Hout, Wout, 1,   C_out)).broadcast_to((N, Hout, Wout, C_in, C_out))

                dW_ij = summation(T_e * G_e, axes=(0, 1, 2))   # (C_in, C_out)
                row.append(dW_ij)
            dW_rows.append(stack(row, axis=0))
        dW = stack(dW_rows, axis=0)  # (K, K, C_in, C_out)

        return dA, dW


def conv_transpose(a, b, stride=1, padding=0):
    return ConvTranspose(stride, padding)(a, b)


class Unbind(TensorTupleOp):
    def __init__(self, axis: int = 0):
        """
        Unbinds (splits) a tensor along an axis into a tuple of tensors,
        removing that axis (like PyTorch's tensor.unbind(dim)).
        
        Parameters:
        axis - dimension to unbind along
        """
        self.axis = axis

    def compute(self, A):
        # Same core logic as Split.compute
        ndim = len(A.shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        n = A.shape[axis]
        parts = []
        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis - 1)
        part_shape = tuple(A.shape[:axis] + A.shape[axis + 1:])
        for i in range(n):
            idx = tuple(left + [slice(i, i + 1)] + right)
            piece = A[idx].compact().reshape(part_shape)
            parts.append(piece)
        return tuple(parts)

    def gradient(self, out_grad, node):
        # Inverse of unbind is stack along the same axis
        return (stack(list(out_grad), self.axis),)


def unbind(a, axis=0):
    return Unbind(axis)(a)


class Sin(TensorOp):
    def compute(self, a):
        return array_api.sin(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        return out_grad * cos(a)


def sin(a):
    return Sin()(a)

class Cos(TensorOp):
    def compute(self, a):
        return array_api.cos(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        # d cos(a) = -sin(a)
        return -out_grad * sin(a)


def cos(a):
    return Cos()(a)


class SiLU(TensorOp):
    def compute(self, a):
        # sigmoid(a) = 1 / (1 + exp(-a))
        return a * ((1.0 + array_api.exp(-a)) ** -1)

    def gradient(self, out_grad, node):
        (a,) = node.inputs

        # compute sigmoid(a) using Needle ops
        sig = 1.0 / (1.0 + exp(-a))

        # silu grad: sig + a * sig * (1 - sig)
        grad = sig + a * sig * (1 - sig)
        return out_grad * grad


def silu(a):
    return SiLU()(a)



class Slice(TensorOp):
    def __init__(self, idx):
        # raw python index or slice/tuple
        self.idx = idx

    def _normalize_index(self, idx, shape):
        """
        Convert Python indexing (with negatives) into proper Needle indexing.
        Supports:
            int
            slice
            tuple[int | slice]
        """
        if isinstance(idx, int):
            # handle negative indexes
            if idx < 0:
                idx += shape[0]
            return idx

        elif isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step

            # Default step = 1
            if step is None:
                step = 1

            # Adjust start
            if start is None:
                start = 0
            elif start < 0:
                start += shape[0]

            # Adjust stop
            if stop is None:
                stop = shape[0]
            elif stop < 0:
                stop += shape[0]

            return slice(start, stop, step)

        elif isinstance(idx, tuple):
            # process each element according to its dimension
            assert len(idx) <= len(shape)
            return tuple(self._normalize_index(i, (shape[d],)) if not isinstance(i, slice)
                         else self._normalize_index(i, (shape[d],))
                         for d, i in enumerate(idx))

        else:
            raise TypeError(f"Unsupported index type {type(idx)}")

    def compute(self, a):
        # Convert Python indexing into explicit normalized slicing
        norm_idx = self._normalize_index(self.idx, a.shape)
        return a[norm_idx]

    def gradient(self, out_grad, node):
        """
        Gradient for slicing is NOT implemented here.
        (As Needle does not have scatter_add yet.)
        """
        raise NotImplementedError("Slice backward not implemented")

        
def slice_tensor(a, idx):
    return Slice(idx)(a)



class Cat(TensorOp):
    def __init__(self, axis: int = 0):
        """
        Concatenates a sequence of arrays along an existing dimension.

        Parameters:
        axis - dimension to concatenate along (can be negative).
        All arrays must have the same shape except in the concatenation axis.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> NDArray:
        ref = args[0]
        ndim = len(ref.shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim

        # Check shapes and compute output shape
        out_shape = list(ref.shape)
        total = 0
        for t in args:
            assert len(t.shape) == ndim, "All tensors must have the same rank"
            for i in range(ndim):
                if i == axis:
                    continue
                assert (
                    t.shape[i] == ref.shape[i]
                ), f"Shape mismatch at dim {i}: {t.shape[i]} vs {ref.shape[i]}"
            total += t.shape[axis]
        out_shape[axis] = total

        out = array_api.empty(tuple(out_shape), device=ref.device)

        # Copy each tensor into the appropriate slice
        start = 0
        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis - 1)
        for t in args:
            size = t.shape[axis]
            end = start + size
            idx = tuple(left + [slice(start, end)] + right)
            out[idx] = t
            start = end

        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        # node.inputs[0] is the TensorTuple that was passed in
        (inputs_tuple,) = node.inputs
        tensors = list(inputs_tuple)
        ndim = len(tensors[0].shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim

        grads = []
        data = out_grad.realize_cached_data()
        start = 0
        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis - 1)

        for t in tensors:
            size = t.shape[axis]
            end = start + size
            idx = tuple(left + [slice(start, end)] + right)
            slice_data = data[idx]
            grads.append(Tensor.make_const(slice_data))
            start = end

        return make_tuple(*grads)


def cat(tensors, dim=0):
    """
    Concatenate a list/tuple of Tensors along an existing axis.
    """
    return Cat(dim)(make_tuple(*tensors))

class Where(TensorOp):
    """
    Needle version of torch.where(cond, x, y)

    cond: boolean Tensor (or 0/1 int/float)
    x, y: Tensors of same or broadcastable shape
    """

    def compute(self, cond, x, y):
        # cond should be boolean or numeric 0/1
        return array_api.where(cond, x, y)

    def gradient(self, out_grad, node):
        cond, x, y = node.inputs

        # Convert cond into a boolean mask NDArray (cached, no grad)
        cond_data = cond.realize_cached_data()
        bool_mask = cond_data.astype(bool)

        # out_grad is a Tensor: convert to NDArray for masking
        og = out_grad.realize_cached_data()

        # Compute grads for x and y
        gx = array_api.where(bool_mask, og, array_api.zeros_like(og))
        gy = array_api.where(bool_mask, array_api.zeros_like(og), og)

        # Wrap masks as const Tensors
        gx = Tensor.make_const(gx)
        gy = Tensor.make_const(gy)

        # No gradient for cond → return None / zero tensor?
        # Following PyTorch: cond is NON-differentiable -> gradient is None.
        return (None, gx, gy)


def where(cond, x, y):
    return Where()(cond, x, y)

class Norm(TensorOp):
    def __init__(self, dim=None, keepdim=False, eps=1e-12):
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

    def compute(self, a):
        # (a * a).sum(dim)
        squared = a * a
        summed = array_api.sum(squared, axis=self.dim, keepdims=self.keepdim)
        return (summed + self.eps) ** 0.5

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        eps = self.eps

        norm = Norm(self.dim, self.keepdim, eps)(a)

        grad = out_grad * a / norm.broadcast_to(a.shape)

        return grad


def norm(a, dim=None, keepdim=False, eps=1e-12):
    return Norm(dim, keepdim, eps)(a)

class Clamp(TensorOp):
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def compute(self, a):
        out = a
        if self.min_val is not None:
            out = array_api.maximum(out, self.min_val)
        if self.max_val is not None:
            out = -array_api.maximum(-out, -self.max_val)
        return out

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        mask = (a >= (self.min_val if self.min_val is not None else a)) * \
               (a <= (self.max_val if self.max_val is not None else a))
        return out_grad * mask


def clamp(a, min_val=None, max_val=None):
    return Clamp(min_val, max_val)(a)


class Abs(TensorOp):
    def compute(self, a):
        return array_api.abs(a)

    def gradient(self, out_grad, node):
        (a,) = node.inputs
        # sign(a) = a>=0? 1 : -1, but 0 → 0 for safety
        sign = where(a > 0, Tensor(1.0), where(a < 0, Tensor(-1.0), Tensor(0.0)))
        return out_grad * sign


def abs(a):
    return Abs()(a)

class Sign(TensorOp):
    def compute(self, a):
        return array_api.where(a > 0, 1.0,
                array_api.where(a < 0, -1.0, 0.0))

    def gradient(self, out_grad, node):
        # derivative of sign(x) is 0 almost everywhere
        return out_grad * 0.0


def sign(a):
    return Sign()(a)

class Chunk(TensorTupleOp):
    def __init__(self, chunks: int, axis: int = 0):
        """
        Split tensor into `chunks` equal parts along `axis`, like torch.chunk
        (we only support exact divisibility, no ragged last chunk).
        """
        self.chunks = chunks
        self.axis = axis

    def compute(self, A):
        ndim = len(A.shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        assert 0 <= axis < ndim, f"Invalid axis {self.axis} for ndim={ndim}"

        dim_size = A.shape[axis]
        if dim_size % self.chunks != 0:
            raise ValueError(
                f"Cannot chunk dimension of size {dim_size} into {self.chunks} equal parts"
            )

        chunk_size = dim_size // self.chunks
        parts = []

        left = [slice(None)] * axis
        right = [slice(None)] * (ndim - axis - 1)

        for i in range(self.chunks):
            start = i * chunk_size
            stop = (i + 1) * chunk_size
            idx = tuple(left + [slice(start, stop)] + right)
            piece = A[idx].compact()
            parts.append(piece)

        return tuple(parts)

    def gradient(self, out_grad, node):
        """
        Inverse of chunk along the same axis is concatenation along that axis.
        out_grad is a TensorTuple.
        """
        return (cat(list(out_grad), dim=self.axis),)


def chunk(a, chunks: int, dim: int = 0):
    return Chunk(chunks, dim)(a)

    