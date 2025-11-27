import needle as nd
from needle import nn, init
from needle.backend_selection import array_api

def naive_attn(q, k, v, scale):
    # q,k,v: (B,H,N,D) Tensors
    # convert to (B,H,N,D) -> (B,H,N,N)
    kt = k.transpose((-2, -1))
    attn = (q * scale) @ kt
    attn = attn.softmax()
    return attn @ v

def flash_attn_wrapper(q, k, v, scale):
    # expect q,k,v on cuda
    q_scaled = q * scale
    # call your flash op: (B,H,N,D)
    from needle.ops import flash_attention
    return flash_attention(q_scaled, k, v)

B, H, N, D = 2, 4, 34, 64
dev = nd.cuda()

q = nd.Tensor(init.randn(B, H, N, D, device=dev), device=dev)
k = nd.Tensor(init.randn(B, H, N, D, device=dev), device=dev)
v = nd.Tensor(init.randn(B, H, N, D, device=dev), device=dev)
scale = D ** -0.5

out_naive = naive_attn(q, k, v, scale)
out_flash = flash_attn_wrapper(q, k, v, scale)

import numpy as np
diff = np.abs((out_naive - out_flash).numpy())
print("max abs diff:", diff.max())
print("mean abs diff:", diff.mean())
