from __future__ import annotations

import math
from math import sqrt
from typing import Literal
from functools import partial

import torch
from torch import nn, pi
from torch.special import expm1
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from x_transformers import Decoder
import sys
sys.path.append('/cpfs01/user/hanyujin/causal-dm/AR_diff/autoregressive-diffusion-pytorch/autoregressive_diffusion_pytorch')
from diffusers import AutoencoderKL
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

# sinusoidal embedding

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
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# simple mlp

class MLP(Module):
    def __init__(
        self,
        dim_cond,
        dim_input,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        cond = F.silu(time_emb + cond)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

        return denoised

# gaussian diffusion

class ElucidatedDiffusion(Module):
    def __init__(
        self,
        dim: int,
        net: MLP,
        device: int,
        *,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        clamp_during_sampling = True,
    
    ):
        super().__init__()

        self.net = net
        self.dim = dim

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.device  = device 
        self.clamp_during_sampling = clamp_during_sampling

    # @property
    # def device(self):
    #     return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_seq, sigma, *, cond, clamp = None):
        clamp = default(clamp, self.clamp_during_sampling)

        batch, _ = noised_seq.shape[0], noised_seq.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = self.device)

        padded_sigma = right_pad_dims_to(noised_seq, sigma)

        net_out = self.net(
            self.c_in(padded_sigma) * noised_seq,
            times = self.c_noise(sigma),
            cond = cond
        )

        out = self.c_skip(padded_sigma) * noised_seq +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, cond, num_sample_steps = None, clamp = None):
        clamp = default(clamp, self.clamp_during_sampling)
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (cond.shape[0], self.dim)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        seq = init_sigma * torch.randn(shape, device = self.device)

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            seq_hat = seq + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(seq_hat, sigma_hat, cond = cond, clamp = clamp)
            denoised_over_sigma = (seq_hat - model_output) / sigma_hat

            seq_next = seq_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(seq_next, sigma_next, cond = cond, clamp = clamp)
                denoised_prime_over_sigma = (seq_next - model_output_next) / sigma_next
                seq_next = seq_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            seq = seq_next

        if clamp:
            seq = seq.clamp(-1., 1.)

        return seq

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, seq, *, cond):
        batch_size, dim, device = *seq.shape, self.device

        assert dim == self.dim, f'dimension of sequence being passed in must be {self.dim} but received {dim}'

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = right_pad_dims_to(seq, sigmas)

        noise = torch.randn_like(seq)

        noised_seq = seq + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_seq, sigmas, cond = cond)

        losses = F.mse_loss(denoised, seq, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        loss_noweight = losses.mean()
        losses = losses * self.loss_weight(sigmas)

        return losses.mean(),loss_noweight

# main model, a decoder with continuous wrapper + small denoising mlp

# class AutoregressiveDiffusion(Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         max_seq_len,
#         device,
#         depth = 8,
#         dim_head = 64,
#         heads = 8,
#         mlp_depth = 3,
#         mlp_width = None,
#         dim_input = None,
#         decoder_kwargs: dict = dict(),
#         mlp_kwargs: dict = dict(),
#         diffusion_kwargs: dict = dict(
#             clamp_during_sampling = True
#         ),
#         use_parallel_order = False
#     ):
#         super().__init__()

#         self.device = device
#         self.start_token = nn.Parameter(torch.zeros(dim, device=self.device))
#         self.max_seq_len = max_seq_len
#         self.abs_pos_emb = nn.Embedding(max_seq_len, dim).to(self.device)

#         dim_input = default(dim_input, dim)
#         self.dim_input = dim_input
#         self.proj_in = nn.Linear(dim_input, dim)

#         self.transformer = Decoder(
#             dim = dim,
#             depth = depth,
#             heads = heads,
#             attn_dim_head = dim_head,
#             **decoder_kwargs
#         ).to(self.device)

#         self.denoiser = MLP(
#             dim_cond = dim,
#             dim_input = dim_input,
#             depth = mlp_depth,
#             width = default(mlp_width, dim),
#             **mlp_kwargs
#         ).to(self.device)

#         self.diffusion = ElucidatedDiffusion(
#             dim_input,
#             self.denoiser,
#             self.device,
#             **diffusion_kwargs,
#         )

#         self.use_parallel_order = use_parallel_order

#     def mask_tokens(self, seq, current_pos):
#         b, seq_len, dim = seq.shape
#         half_len = seq_len // 2
        
#         if current_pos >= half_len:
#             # 如果当前位置在后半部分，屏蔽前半部分
#             mask = torch.zeros_like(seq)
#             mask[:, half_len:current_pos+1, :] = 1
#         else:
#             # 如果当前位置在前半部分，不进行屏蔽
#             mask = torch.ones_like(seq)
        
#         return seq * mask

#     def forward(self, seq):

#         b, seq_len, dim = seq.shape
#         # print('seq_len:',seq_len,'self.max_seq_len:',self.max_seq_len)
#         # print('dim:',dim,'self.dim_input:',self.dim_input)
#         assert dim == self.dim_input
#         assert seq_len == self.max_seq_len
#         seq, target = seq[:, :-1], seq

#         # append start tokens
#         seq = self.proj_in(seq)
#         start_token = repeat(self.start_token, 'd -> b 1 d', b = b)

#         seq = torch.cat((start_token, seq), dim = 1).to(self.device)
#         seq = seq + self.abs_pos_emb(torch.arange(seq_len, device = self.device))

#         cond = []
#         for i in range(seq_len):
#             if self.use_parallel_order:
#                 masked_seq = self.mask_tokens(seq[:, :i+1], i)
#             else:
#                 masked_seq = seq[:, :i+1]
            
#             cond_i = self.transformer(masked_seq)
#             cond.append(cond_i[:, -1:])
        
#         cond = torch.cat(cond, dim=1)

#         # pack batch and sequence dimensions, so to train each token with different noise levels
#         target, _ = pack_one(target, '* d')
#         cond, _ = pack_one(cond, '* d')

#         diffusion_loss, diffusion_loss_noweight = self.diffusion(target, cond = cond)

#         return diffusion_loss, diffusion_loss_noweight

#     @torch.no_grad()
#     def sample(self, batch_size = 1, prompt = None):
#         self.eval()

#         start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch_size)

#         if not exists(prompt):
#             out = torch.empty((batch_size, 0, self.dim_input), device = self.device, dtype = torch.float32)
#         else:
#             out = prompt

#         for i in tqdm(range(self.max_seq_len), desc = 'tokens'):
#             cond = self.proj_in(out)
#             cond = torch.cat((start_tokens, cond), dim = 1)
#             cond = cond + self.abs_pos_emb(torch.arange(cond.shape[1], device = self.device))

#             if self.use_parallel_order:
#                 cond = self.mask_tokens(cond, i)

#             cond = self.transformer(cond)
#             last_cond = cond[:, -1]

#             denoised_pred = self.diffusion.sample(cond = last_cond)

#             denoised_pred = rearrange(denoised_pred, 'b d -> b 1 d')
#             out = torch.cat((out, denoised_pred), dim = 1)

#         return out

class AutoregressiveDiffusion(Module):
    def __init__(
        self,
        dim,
        *,
        max_seq_len=2,  # 两个特征 x1 和 x2
        device,
        depth=4,  # 减小模型深度
        dim_head=64,
        heads=8,
        mlp_depth=4,
        mlp_width=None,
        dim_input=1,
        decoder_kwargs: dict = dict(),
        mlp_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(
            clamp_during_sampling=True
        ),
        use_parallel_order=False,
    ):
        super().__init__()

        self.device = device
        self.start_token = nn.Parameter(torch.zeros(dim, device=self.device))
        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim).to(self.device)

        dim_input = default(dim_input, dim)
        self.dim_input = dim_input
        self.proj_in = nn.Linear(dim_input, dim)

        # Transformer Decoder
        self.transformer = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dim_head=dim_head,
            **decoder_kwargs
        ).to(self.device)

        # Denoising MLP
        self.denoiser = MLP(
            dim_cond=dim,
            dim_input=dim_input,
            depth=mlp_depth,
            width=default(mlp_width, dim),
            **mlp_kwargs
        ).to(self.device)

        # Diffusion module
        self.diffusion = ElucidatedDiffusion(
            dim_input,
            self.denoiser,
            self.device,
            **diffusion_kwargs,
        )

        self.use_parallel_order = use_parallel_order

    def mask_tokens(self, seq, current_pos):
        """
        Masks tokens to ensure autoregressive behavior.
        """
        b, seq_len, dim = seq.shape
        mask = torch.ones_like(seq)
        mask[:, current_pos+1:, :] = 0  # Mask future tokens
        return seq * mask

    def forward(self, seq):
        """
        Forward pass for 1D data (x1 -> x2).
        Args:
            seq: Input sequence of shape [B, 2, 1].
        Returns:
            diffusion_loss, diffusion_loss_noweight: Loss values.
        """
        # print("seq",seq)
        # print("seq.shape:",seq.shape)
        b, seq_len, dim = seq.shape
        assert dim == self.dim_input
        assert seq_len == self.max_seq_len

        # Split input into x1 and x2
        x1, x2 = seq[:, 0:1, :], seq[:, 1:2, :]  # Separate x1 and x2

        # Process x1
        x1_proj = self.proj_in(x1)
        start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
        x1_cond = torch.cat((start_token, x1_proj), dim=1)
        x1_cond = x1_cond + self.abs_pos_emb(torch.arange(x1_cond.shape[1], device=self.device))
        x1_cond = self.transformer(x1_cond)
        x1_output = x1_cond[:, -1:]  # Last token represents x1 output

        # Process x2 conditioned on x1
        x2_cond_input = torch.cat((x1_output, self.proj_in(x2)), dim=1)
        x2_cond_input = x2_cond_input + self.abs_pos_emb(torch.arange(x2_cond_input.shape[1], device=self.device))
        x2_cond = self.transformer(x2_cond_input)

        # Pack and calculate diffusion loss
        target, _ = pack_one(seq, '* d')
        cond, _ = pack_one(x2_cond, '* d')
        diffusion_loss, diffusion_loss_noweight = self.diffusion(target, cond=cond)

        return diffusion_loss, diffusion_loss_noweight

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Sampling for 1D data (x1 -> x2).
        """
        self.eval()
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b=batch_size)
        out = torch.empty((batch_size, 0, self.dim_input), device=self.device, dtype=torch.float32)

        for i in range(self.max_seq_len):
            cond = self.proj_in(out)
            cond = torch.cat((start_tokens, cond), dim=1)
            cond = cond + self.abs_pos_emb(torch.arange(cond.shape[1], device=self.device))

            if self.use_parallel_order:
                cond = self.mask_tokens(cond, i)

            cond = self.transformer(cond)
            last_cond = cond[:, -1]

            denoised_pred = self.diffusion.sample(cond=last_cond)
            denoised_pred = rearrange(denoised_pred, 'b d -> b 1 d')
            out = torch.cat((out, denoised_pred), dim=1)

        return out


# image wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class ImageAutoregressiveDiffusion(Module):
    def __init__(
        self,
        *,
        device,
        image_size=2,
        patch_size=1,
        channels=1,
        kl_path=None,  # 预训练 AutoencoderKL 的路径
        model: dict = dict(),
        use_parallel_order=False,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device

        assert (image_size % patch_size == 0), "image_size 必须能被 patch_size 整除"
        
        # 针对像素空间的 token 分块
        num_patches = (image_size // patch_size) 
        dim_input = patch_size
        
        # 序列长度和特征维度
        self.num_patches = num_patches
        self.dim_input = dim_input

        # 定义用于像素 token 的 re-arrange 操作
        self.to_tokens = lambda x: x.unsqueeze(1)

        # 自回归扩散模型
        self.model = AutoregressiveDiffusion(
            **model,
            dim_input=dim_input,         
            max_seq_len=num_patches,   
            use_parallel_order=use_parallel_order,
            device = device
        )

        # 定义用于像素还原的 re-arrange 操作
        self.to_image = lambda x: x.squeeze(1)
       
    def forward(self, inputs):
        """
        Forward pass for 1D input (x1 -> x2).
        Args:
            inputs: torch.Size([B, 2]) with features [x1, x2].
        Returns:
            diffusion_loss, diffusion_loss_noweight: Loss values.
        """
        # 分离 x1 和 x2
        x1, x2 = inputs[:, 0:1], inputs[:, 1:2]  # [B, 1] for each feature
        
        # 转换为 tokens
        x1_tokens = self.to_tokens(x1)  # [B, 1, 1]
        x2_tokens = self.to_tokens(x2)  # [B, 1, 1]
        
        # 拼接 x1 和 x2 tokens，模拟条件生成
        tokens = torch.cat([x1_tokens, x2_tokens], dim=1)  # [B, 2, 1]

        # 扩散模型的前向计算
        diffusion_loss, diffusion_loss_noweight = self.model(tokens)
        return diffusion_loss, diffusion_loss_noweight

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Sampling for 1D input.
        Args:
            batch_size: Number of samples to generate.
        Returns:
            samples: Generated data of size [batch_size, 2].
        """
        # 从扩散模型中生成标准化的 tokens
        latent_tokens = self.model.sample(batch_size=batch_size)  # [B, 2, 1]

        # 将 tokens 转回输入格式
        samples = self.to_image(latent_tokens)  # [B, 2]
        print("samples:",samples)
        print("samples.shape:",samples.shape)
        return samples


