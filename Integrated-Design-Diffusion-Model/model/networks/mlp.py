from __future__ import annotations
import torch
from torch import nn, pi
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, pack, unpack


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class LearnedSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        if x.ndim == 1:  # 只有一个维度 (batch_size,)
            x = rearrange(x, 'b -> b 1')  # 添加第二个维度
        elif x.ndim == 2:  # 已经是 (batch_size, 1)
            pass 
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class MLP(Module):
    def __init__(
        self,
        dim_input,
        depth=3,
        width=1024,
        dropout=0.,
        device = 'cpu'
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.dropout = dropout
        
        # Time embedding
        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_input),
            nn.Linear(dim_input + 1, dim_input),
        )
        self.to(device)
        # MLP layers
        layers = ModuleList([])
        for _ in range(depth):
            adaptive_layernorm = AdaptiveLayerNorm(dim_input, dim_condition=dim_input)
            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )
            block_out_gamma = nn.Linear(dim_input, dim_input, bias=False)
            nn.init.zeros_(block_out_gamma.weight)
            layers.append(ModuleList([adaptive_layernorm, block, block_out_gamma]))
        self.layers = layers

    def forward(self, noised, times,y=None):
        assert noised.ndim == 4  # (batch, dim_input)
        batch_size = noised.shape[0]
        noised = noised.view(batch_size, -1)
        # Process time embedding
        time_emb = self.to_time_emb(times.unsqueeze(-1).float())  # (batch, dim_input)
        
        denoised = noised
        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition=time_emb)
            block_out = block(denoised) * (block_out_gamma(time_emb) + 1.)
            denoised = block_out + residual
        image_size = int((denoised.shape[1] // 3) ** 0.5)
        return denoised.view(denoised.shape[0], 3, image_size, image_size)
