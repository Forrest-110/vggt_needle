# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



from functools import partial
from typing import Callable
import collections
from itertools import repeat


from needle import Tensor, nn, init, ops

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)



class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = nn.MultiheadAttention,
        mlp_ratio=4.0,
        **block_kwargs,
    ):
        """
        Self attention block
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn = attn_class(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, **block_kwargs)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)

    def forward(self, x, mask=None):
        # Prepare the mask for PyTorch's attention (it expects a different format)
        # attn_mask = mask if mask is not None else None
        # Normalize before attention
        x = self.norm1(x)

        # PyTorch's MultiheadAttention returns attn_output, attn_output_weights
        # attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)

        attn_output, _ = self.attn(x, x, x)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        """
        Cross attention block
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)

    def forward(self, x, context, mask=None):
        # Normalize inputs
        x = self.norm1(x)
        context = self.norm_context(context)

        # Apply cross attention
        # Note: nn.MultiheadAttention returns attn_output, attn_output_weights
        assert mask is None
        attn_output, _ = self.cross_attn(x, context, context)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x