"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


def _named_tensors(obj: object, prefix: str, out: dict[str, Tensor], seen_buffers: set[int]):
    """
    Collect (name -> Tensor) for Parameters and buffers in a tree of Modules / dicts / lists.

    - prefix: hierarchical name (e.g. "layer1.0")
    - out: dict being filled
    - seen_buffers: ids of buffer tensors, so we don't double-count them
    """
    if isinstance(obj, Module):
        # First: parameters + buffers in this module's __dict__
        for name, value in obj.__dict__.items():
            if name.startswith("_"):
                # skip internal attributes like _buffers, etc.
                continue

            full_name = name if prefix == "" else f"{prefix}.{name}"

            # Parameters
            if isinstance(value, Parameter):
                out[full_name] = value

            # Buffers (registered via register_buffer)
            elif name in obj._buffers:
                buf = obj._buffers[name]
                if buf is not None and id(buf) not in seen_buffers:
                    out[full_name] = buf
                    seen_buffers.add(id(buf))

            # Recurse into child Modules / containers
            if isinstance(value, Module) or isinstance(value, (dict, list, tuple)):
                _named_tensors(value, full_name, out, seen_buffers)

    elif isinstance(obj, dict):
        for key, value in obj.items():
            full_name = key if prefix == "" else f"{prefix}.{key}"
            if isinstance(value, (Module, dict, list, tuple)):
                _named_tensors(value, full_name, out, seen_buffers)
            elif isinstance(value, Parameter) or isinstance(value, Tensor):
                out[full_name] = value

    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            full_name = str(idx) if prefix == "" else f"{prefix}.{idx}"
            if isinstance(value, (Module, dict, list, tuple)):
                _named_tensors(value, full_name, out, seen_buffers)
            elif isinstance(value, Parameter) or isinstance(value, Tensor):
                out[full_name] = value


class Module:
    def __init__(self) -> None:
        self.training = True
        self._buffers = {}   # name -> Tensor or None

    def register_buffer(self, name: str, tensor):
        """
        Register a persistent buffer (non-parameter tensor).

        Behaves like PyTorch:
          - tensor may be a Tensor or None
          - buffer is stored and returned in module.state_dict() style
          - buffer is not considered a parameter
        """
        if hasattr(self, name):
            raise KeyError(f"Attribute '{name}' already exists in the module")
        if not (tensor is None or isinstance(tensor, Tensor)):
            raise TypeError("register_buffer expects a Tensor or None")

        self._buffers[name] = tensor
        setattr(self, name, tensor)
        return tensor

    def buffers(self):
        """Return a list of all buffers in the module and its children."""
        bufs = list(self._buffers.values())
        for child in self._children():
            bufs.extend(child.buffers())
        return bufs

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        # exclude buffers
        params = _unpack_params(self.__dict__)
        return [p for p in params if p not in self._buffers.values()]

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ======================================================
    #  NEW: state_dict / load_state_dict
    # ======================================================

    def state_dict(self) -> dict[str, Tensor]:
        """
        Return a flat dict mapping names to Tensors (parameters + buffers),
        similar to PyTorch's state_dict.
        """
        out: dict[str, Tensor] = {}
        seen_buffers: set[int] = set()
        _named_tensors(self, prefix="", out=out, seen_buffers=seen_buffers)
        return out

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True):
        """
        Load parameters and buffers from a state_dict produced by state_dict().

        Args:
            state_dict: dict mapping names to Tensors
            strict: if True, raise if there are missing/unexpected keys

        Returns:
            (missing_keys, unexpected_keys) for inspection if needed.
        """
        # Current model state (for existence / shape checks)
        own_state = self.state_dict()

        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []

        # First, load the keys we know about
        for name, tensor in state_dict.items():
            if name not in own_state:
                # Parameter/buffer in checkpoint not found in this model
                unexpected_keys.append(name)
                continue

            cur = own_state[name]
            if hasattr(cur, "shape") and hasattr(tensor, "shape"):
                if cur.shape != tensor.shape:
                    shape_mismatch_keys.append(name)
                    continue

            # Rebind the attribute in the module tree
            self._set_by_name(name, tensor)

        # Then, if strict, check for keys present in model but missing in checkpoint
        for name in own_state.keys():
            if name not in state_dict:
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys or shape_mismatch_keys):
            msg = []
            if missing_keys:
                msg.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                msg.append(f"Unexpected keys: {unexpected_keys}")
            if shape_mismatch_keys:
                msg.append(f"Shape mismatch for keys: {shape_mismatch_keys}")
            raise KeyError("Error(s) in load_state_dict:\n  " + "\n  ".join(msg))

        return missing_keys, unexpected_keys

    def _set_by_name(self, name: str, value: Tensor):
        """
        Helper for load_state_dict: set a parameter/buffer by its dotted name.
        E.g. "layer1.blocks.0.weight"
        """
        parts = name.split(".")
        obj = self
        # Traverse to the parent of the leaf
        for p in parts[:-1]:
            if isinstance(obj, (list, tuple)) and p.isdigit():
                obj = obj[int(p)]
            elif isinstance(obj, dict):
                obj = obj[p]
            else:
                obj = getattr(obj, p)
        last = parts[-1]

        if isinstance(obj, (list, tuple)) and last.isdigit():
            # list/tuple item
            idx = int(last)
            # For tuple this would error, but we assume lists for containers.
            obj[idx] = value
        elif isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is None:
            modules = []
        assert isinstance(modules, (list, tuple))

        self._modules_list = []
        self.extend(modules)

    def __len__(self):
        return len(self._modules_list)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __setitem__(self, idx, module):
        self._modules_list[idx] = module
        setattr(self, str(idx), module)

    def append(self, module):
        idx = len(self._modules_list)
        self._modules_list.append(module)
        setattr(self, str(idx), module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def insert(self, index, module):
        self._modules_list.insert(index, module)

        # Renumber all attributes
        for i, m in enumerate(self._modules_list):
            setattr(self, str(i), m)

class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.use_bias = bias
        w = init.kaiming_uniform(
            out_features, in_features, device=device, dtype=dtype, requires_grad=True
        )
        self.weight = Parameter(w)
        if bias:
            b = init.kaiming_uniform(
                out_features, 1, device=device, dtype=dtype, requires_grad=True
            )
            b = ops.reshape(b, (1, out_features))
            self.bias = Parameter(b)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        
        Wt = ops.transpose(self.weight, axes=(-1, -2)) + 0.0  # (in_features, out_features)
        out = ops.matmul(X, Wt)
        # print(X.device)
        # exit()
        if self.bias is not None:
            b_row = self.bias
            b_full = ops.broadcast_to(b_row, out.shape)
            out = out + b_full
        return out


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        b = X.shape[0]
        dim = 1
        for s in X.shape[1:]:
            dim *= s
        return ops.reshape(X, (b, dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.silu(x)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # tanh approximation:
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
        c = 0.044715
        sqrt_2_over_pi = 0.7978845608028654  # = sqrt(2/pi)

        x3 = ops.multiply(x, ops.multiply(x, x))
        inner = sqrt_2_over_pi * ops.add(x, c * x3)
        return ops.multiply(
            0.5 * x,
            1.0 + ops.tanh(inner)
        )

class Sequential(Module):
    """
    PyTorch-like Sequential container.

    Children are stored as attributes "0", "1", "2", ...
    so that state_dict keys look like:

        model.heads.0.weight
        model.heads.1.bias

    and not "...heads.modules.0.weight".
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._modules_list = []
        for idx, m in enumerate(modules):
            self._modules_list.append(m)
            # register as attribute so Module/state_dict can see it
            setattr(self, str(idx), m)

    def __len__(self):
        return len(self._modules_list)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __setitem__(self, idx, module: Module):
        self._modules_list[idx] = module
        setattr(self, str(idx), module)

    def append(self, module: Module):
        """Append a new module at the end, like list.append."""
        idx = len(self._modules_list)
        self._modules_list.append(module)
        setattr(self, str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for m in self._modules_list:
            out = m(out)
        return out


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        N, C = logits.shape
        y_onehot = init.one_hot(
            C, y, device=logits.device, dtype=logits.dtype, requires_grad=False
        )
        z_y = ops.summation(ops.multiply(logits, y_onehot), axes=(-1,))  # (N,)
        loss = ops.logsumexp(logits, axes=(-1,)) - z_y                                               # (N,)
        return ops.summation(loss) / N


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )

        self.running_mean = Tensor(
            init.zeros(dim, device=device, dtype=dtype).cached_data,
            device=device, dtype=dtype, requires_grad=False,
        )
        self.running_var = Tensor(
            init.ones(dim, device=device, dtype=dtype).cached_data,
            device=device, dtype=dtype, requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]

        if self.training:
            mean = ops.summation(x, axes=(0,)) / N 
            m_b = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)

            xc = x - m_b
            var = ops.summation(ops.multiply(xc, xc), axes=(0,)) / N 
            v_b = ops.broadcast_to(ops.reshape(var, (1, self.dim)), x.shape)

            rm = self.running_mean.cached_data
            rv = self.running_var.cached_data
            bm = mean.cached_data
            bv = var.cached_data
            new_rm = (1.0 - self.momentum) * rm + self.momentum * bm
            new_rv = (1.0 - self.momentum) * rv + self.momentum * bv
            self.running_mean = Tensor(new_rm, device=x.device, dtype=x.dtype, requires_grad=False)
            self.running_var  = Tensor(new_rv, device=x.device, dtype=x.dtype, requires_grad=False)

            std = ops.power_scalar(v_b + self.eps, 0.5)
            x = xc / std
        else:
            m_b = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
            v_b  = ops.broadcast_to(ops.reshape(self.running_var,  (1, self.dim)), x.shape)
            x = (x - m_b) / ops.power_scalar(v_b + self.eps, 0.5)

        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias,   (1, self.dim)), x.shape)
        return ops.multiply(w, x) + b



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=(-1,)) / self.dim
        mean = ops.reshape(mean, (x.shape[0], 1))

        x_c = x - ops.broadcast_to(mean, x.shape)
        var = ops.summation(ops.multiply(x_c, x_c), axes=(-1,)) / self.dim
        var = ops.reshape(var, (x.shape[0], 1))
        std = ops.power_scalar(var + self.eps, 0.5)
        x_n = x_c / ops.broadcast_to(std, x.shape)

        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b   = ops.broadcast_to(ops.reshape(self.bias,  (1, self.dim)), x.shape)
        return ops.multiply(w, x_n) + b


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = init.randb(
            *x.shape, p=keep_prob,
            device=x.device, dtype="float32", requires_grad=False
        )

        return ops.multiply(x, mask) / keep_prob


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))



class LayerNorm(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        """
        Needle implementation of LayerNorm with optional elementwise affine.

        Args:
            dim: normalized dimension (last dimension)
            eps: numerical stability
            elementwise_affine: if True, learn weight & bias (PyTorch default)
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(
                init.ones(dim, device=device, dtype=dtype, requires_grad=True)
            )
            self.bias = Parameter(
                init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
            )
        else:
            # Register as buffers to mimic PyTorch (not trainable)
            self.register_buffer("weight", Tensor(init.ones(dim, device=device, dtype=dtype), requires_grad=False))
            self.register_buffer("bias",   Tensor(init.zeros(dim, device=device, dtype=dtype), requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (..., dim)  
        Normalize over the last dimension exactly like nn.LayerNorm.
        """
        assert x.shape[-1] == self.dim, (
            f"Expected last dimension {self.dim}, got {x.shape[-1]}"
        )

        D = x.shape[-1]

        # mean (...,1)
        mean = ops.summation(x, axes=(-1,)) / D
        mean = ops.reshape(mean, x.shape[:-1] + (1,))

        x_c = x - ops.broadcast_to(mean, x.shape)

        # variance (...,1)
        var = ops.summation(ops.multiply(x_c, x_c), axes=(-1,)) / D
        var = ops.reshape(var, x.shape[:-1] + (1,))

        # std
        std = ops.power_scalar(var + self.eps, 0.5)

        x_n = x_c / ops.broadcast_to(std, x.shape)

        # if disabled, just return normalization
        if not self.elementwise_affine:
            return x_n

        # Apply affine: weight, bias of shape (dim,)
        ndim = len(x.shape)
        w = ops.reshape(self.weight, (1,) * (ndim - 1) + (self.dim,))
        b = ops.reshape(self.bias,  (1,) * (ndim - 1) + (self.dim,))

        w = ops.broadcast_to(w, x.shape)
        b = ops.broadcast_to(b, x.shape)

        return ops.multiply(w, x_n) + b



class MultiheadAttention(Module):
    """
    Needle version of nn.MultiheadAttention (simplified, PyTorch-compatible).

    - batch_first only (inputs are (B, T, E))
    - no attention mask / key padding mask support yet
    - no dropout inside attention (you can wrap externally)
    - uses in_proj_weight / in_proj_bias like torch.nn.MultiheadAttention

    Args:
        embed_dim: total embedding dimension (E)
        num_heads: number of attention heads (H)
        bias: whether to use bias in projection layers
        batch_first: must be True (enforced)
        device, dtype: passed to internal parameters

    Forward:
        query: (B, T_q, E)
        key:   (B, T_k, E)
        value: (B, T_k, E)

    Returns:
        attn_output:  (B, T_q, E)
        attn_weights: (B, num_heads, T_q, T_k)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
        device=None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        assert batch_first, "This MultiheadAttention only supports batch_first=True for now."
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.bias = bias

        # ------------------------------------------------------------------
        # Match torch.nn.MultiheadAttention parameter naming & shapes:
        #
        #   in_proj_weight: (3*E, E)
        #   in_proj_bias:   (3*E,)
        #
        # These pack Q, K, V projections together.
        # ------------------------------------------------------------------

        # Use torch.empty as requested, then wrap into Parameter (Needle Tensor)
        in_w = init.rand(3 * embed_dim, embed_dim)
        self.in_proj_weight = Parameter(
            in_w
        )

        if bias:
            in_b = init.rand(3 * embed_dim)
            self.in_proj_bias = Parameter(
                in_b.detach()
            )
        else:
            self.in_proj_bias = None

        # Output projection stays as a standard Linear
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

        # You probably want to initialize weights properly; you can add
        # a custom reset_parameters() here if you like.

    def _shape_proj(self, x: Tensor, B: int, T: int) -> Tensor:
        """
        Projected tensor x of shape (B, T, E) -> (B, num_heads, T, head_dim)
        """
        # (B, T, E) -> (B, T, H, D)
        x = x.reshape((B, T, self.num_heads, self.head_dim))
        # -> (B, H, T, D)
        x = x.permute((0, 2, 1, 3))
        return x

    def _linear(self, x: Tensor, w: Tensor, b: Tensor | None) -> Tensor:
        """
        x: (B, T, E)
        w: (E_out, E_in)  [slice of in_proj_weight, shape (E, E)]
        b: (E_out,)
        We implement y = x @ w^T + b, like torch.nn.functional.linear.
        """
        # w_T: (E_in, E_out)
        w_T = ops.transpose(w, (1, 0))+0.0
        y = ops.matmul(x, w_T)  # (B, T, E_out)
        if b is not None:
            b = b+0.0
            b_b = b.reshape((1, 1, -1)).broadcast_to(y.shape)
            y = y + b_b
        return y

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = True,
    ):
        """
        Args:
            query: (B, T_q, E)
            key:   (B, T_k, E)
            value: (B, T_k, E)

        Returns:
            attn_output:  (B, T_q, E)
            attn_weights: (B, num_heads, T_q, T_k) or None
        """
        assert query.ndim == 3 and key.ndim == 3 and value.ndim == 3, \
            "Expected (B, T, E) for query/key/value."

        B, T_q, _ = query.shape
        Bk, T_k, _ = key.shape
        Bv, T_v, _ = value.shape
        assert B == Bk == Bv, "Batch sizes of query/key/value must match."
        assert T_k == T_v, "Key and value sequence lengths must match."
        assert query.shape[2] == self.embed_dim
        assert key.shape[2] == self.embed_dim
        assert value.shape[2] == self.embed_dim

        E = self.embed_dim
        W = self.in_proj_weight   # (3E, E)
        b = self.in_proj_bias     # (3E,) or None

        # Slice weights/biases for Q, K, V
        W_q = W[:E, :]            # (E, E)
        W_k = W[E:2*E, :]         # (E, E)
        W_v = W[2*E:3*E, :]       # (E, E)

        if b is not None:
            b_q = b[:E]           # (E,)
            b_k = b[E:2*E]
            b_v = b[2*E:3*E]
        else:
            b_q = b_k = b_v = None

        # 1) Linear projections using packed weights
        # (B, T_q, E)  /  (B, T_k, E)
        q = self._linear(query, W_q, b_q)
        k = self._linear(key,   W_k, b_k)
        v = self._linear(value, W_v, b_v)

        # 2) Reshape to multi-head
        # (B, H, T, D)
        q = self._shape_proj(q, B, T_q)
        k = self._shape_proj(k, B, T_k)
        v = self._shape_proj(v, B, T_k)

        # 3) Scaled dot-product attention
        # attn_scores: (B, H, T_q, T_k)
        # q: (B, H, T_q, D), k: (B, H, T_k, D)
        k_t = k.permute((0, 1, 3, 2)) + 0.0  # (B, H, D, T_k)
        attn_scores = ops.matmul(q + 0.0, k_t)  # (B, H, T_q, T_k)

        scale = (self.head_dim ** -0.5)
        attn_scores = attn_scores * scale

        # softmax over key dimension (last axis)
        attn_weights = ops.softmax(attn_scores)  # (B, H, T_q, T_k)

        # 4) Attention output
        # (B, H, T_q, D) = (B,H,T_q,T_k) @ (B,H,T_k,D)
        attn_output_heads = ops.matmul(attn_weights, v + 0.0)  # (B, H, T_q, D)

        # 5) Concatenate heads
        # (B, H, T_q, D) -> (B, T_q, H, D) -> (B, T_q, E)
        attn_output = attn_output_heads.permute((0, 2, 1, 3)) + 0.0
        attn_output = attn_output.reshape((B, T_q, self.embed_dim))

        # 6) Final output projection
        attn_output = self.out_proj(attn_output)  # (B, T_q, E)

        if not need_weights:
            return attn_output, None

        return attn_output, attn_weights


class GroupNorm(Module):
    """
    Needle version of nn.GroupNorm.

    Assumes input in NCHW format (N, C, H, W) or more generally (N, C, *spatial).

    Args:
        num_groups (int): number of groups to divide the channels into.
        num_channels (int): total number of channels C.
        eps (float): small constant for numerical stability.
        affine (bool): if True, use learnable per-channel scale and bias.
        device, dtype: passed to parameter initializers.

    Behavior matches PyTorch's GroupNorm:
      - For each sample n and each group g, normalize over all
        elements in that group: C_group * H * W.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Any = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.eps = eps
        self.affine = affine

        if affine:
            # Per-channel gamma/beta, like PyTorch
            self.weight = Parameter(
                init.ones(num_channels, device=device, dtype=dtype, requires_grad=True)
            )
            self.bias = Parameter(
                init.zeros(num_channels, device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, C, *spatial)
        """
        assert x.shape[1] == self.num_channels, \
            f"Expected C={self.num_channels}, got {x.shape[1]}"

        N, C = x.shape[0], x.shape[1]
        spatial_shape = x.shape[2:]  # could be (H, W) or more dims

        # Reshape to (N, G, Cg, *spatial)
        new_shape = (N, self.num_groups, self.channels_per_group, *spatial_shape)
        x_grouped = x.reshape(new_shape)

        # Compute mean and var over (Cg, *spatial) for each (N, G)
        # axes = all dims from 2 onward
        ndim_grouped = len(new_shape)
        reduce_axes = tuple(range(2, ndim_grouped))

        # Number of elements per group for unbiased mean/var
        group_size = 1
        for ax in reduce_axes:
            group_size *= new_shape[ax]

        # mean: (N, G)
        mean = ops.summation(x_grouped, axes=reduce_axes) / float(group_size)

        # Broadcast mean back to x_grouped shape
        mean_shape = (N, self.num_groups) + (1,) * (ndim_grouped - 2)
        mean_b = mean.reshape(mean_shape).broadcast_to(new_shape)

        # variance: E[(x - mean)^2]
        x_centered = x_grouped - mean_b
        var = ops.summation(x_centered * x_centered, axes=reduce_axes) / float(group_size)  # (N, G)

        var_b = var.reshape(mean_shape).broadcast_to(new_shape)

        # Normalize
        std_b = (var_b + self.eps) ** 0.5
        x_norm = x_centered / std_b  # (N, G, Cg, *spatial)

        # Reshape back to (N, C, *spatial)
        x_norm = x_norm.reshape((N, C, *spatial_shape))

        # Apply affine if enabled
        if self.affine:
            # weight, bias: (C,)
            # reshape to (1, C, 1, 1, ...) for broadcasting
            expand_shape = (1, self.num_channels) + (1,) * (len(spatial_shape))
            w = self.weight.reshape(expand_shape).broadcast_to(x_norm.shape)
            b = self.bias.reshape(expand_shape).broadcast_to(x_norm.shape)
            x_norm = x_norm * w + b

        return x_norm