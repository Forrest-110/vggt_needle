import math
from .init_basic import *
from typing import Any

def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p

def _compute_fans_from_shape(shape):
    if len(shape) == 1:
        fan_in = fan_out = int(shape[0])
    elif len(shape) == 2:
        fan_in, fan_out = int(shape[0]), int(shape[1])
    else:
        rf = _prod(shape[:-2])  
        c_in, c_out = int(shape[-2]), int(shape[-1])
        fan_in, fan_out = c_in * rf, c_out * rf
    return fan_in, fan_out


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    *,
    shape = None,
    **kwargs: Any,
):
    if shape is not None:
        fan_in, fan_out = _compute_fans_from_shape(shape)
        a = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return rand(*shape, low=-a, high=a, **kwargs)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(
    fan_in: int,
    fan_out: int,
    gain: float = 1.0,
    *,
    shape = None,
    **kwargs: Any,
):
    if shape is not None:
        fan_in, fan_out = _compute_fans_from_shape(shape)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return randn(*shape, mean=0.0, std=std, **kwargs)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)


def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    *,
    shape = None,
    **kwargs: Any,
):
    assert nonlinearity == "relu", "Only relu supported currently"
    if shape is not None:
        fan_in, _ = _compute_fans_from_shape(shape)
        a = math.sqrt(6.0 / fan_in)
        return rand(*shape, low=-a, high=a, **kwargs)
    a = math.sqrt(6.0 / fan_in)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def kaiming_normal(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    *,
    shape = None,
    **kwargs: Any,
):
    assert nonlinearity == "relu", "Only relu supported currently"
    if shape is not None:
        fan_in, _ = _compute_fans_from_shape(shape)
        std = math.sqrt(2.0 / fan_in)
        return randn(*shape, mean=0.0, std=std, **kwargs)
    std = math.sqrt(2.0 / fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)