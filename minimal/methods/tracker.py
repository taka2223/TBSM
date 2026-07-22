"""Conditional scattering tracker used by each representation field."""

import math

import torch
import torch.nn as nn


class ParameterFreeRMSNorm(nn.Module):
    """RMS normalization without learned affine parameters."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""

    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
        / half_dim
    )
    args = timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _sincos_embedding_1d(pos, dim, max_period=10000):
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=pos.device)
        / half_dim
    )
    args = pos.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SpatialTransformerBlock(nn.Module):
    """Self-attention and MLP block for spatial feature tokens."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = ParameterFreeRMSNorm()
        self.qkv = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim * 3, bias=False))
        self.proj = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.norm2 = ParameterFreeRMSNorm()
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim, mlp_dim, bias=False)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(mlp_dim, hidden_dim, bias=False)),
        )

    def forward(self, x):
        hidden = self.norm1(x)
        batch_size, num_tokens, channels = hidden.shape
        query_key_value = self.qkv(hidden).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = query_key_value.permute(2, 0, 3, 1, 4)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        hidden = (attention @ value).transpose(1, 2).reshape(
            batch_size, num_tokens, channels
        )
        x = x + self.proj(hidden)
        x = x + self.mlp(self.norm2(x))
        return x


class Tracker(nn.Module):
    """Estimate the query-conditioned tracked scattering vector."""

    def __init__(
        self,
        num_classes,
        feat_dim,
        hidden_dim=736,
        time_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        use_potential_grad=False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.use_potential_grad = bool(use_potential_grad)
        self.register_buffer("_pos_embed_cache", torch.empty(0), persistent=False)
        self._pos_embed_cache_key = None

        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(time_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )

        self.x_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList([
            SpatialTransformerBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        if self.use_potential_grad:
            self.refine = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1, bias=False)),
            )
        else:
            self.refine = None

    def _position_embedding(self, N, device, dtype):
        num_tokens = N
        key = (num_tokens, device.type, device.index, dtype)
        if self._pos_embed_cache_key == key and self._pos_embed_cache.numel() > 0:
            return self._pos_embed_cache

        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            )
            pos_y = grid_y.reshape(-1)
            pos_x = grid_x.reshape(-1)
            dim_y = self.hidden_dim // 2
            dim_x = self.hidden_dim - dim_y
            pos_embed = torch.cat([
                _sincos_embedding_1d(pos_y, dim_y),
                _sincos_embedding_1d(pos_x, dim_x),
            ], dim=-1)
        else:
            pos = torch.arange(num_tokens, device=device)
            pos_embed = _sincos_embedding_1d(pos, self.hidden_dim)

        pos_embed = pos_embed.to(dtype=dtype).unsqueeze(0)
        self._pos_embed_cache = pos_embed
        self._pos_embed_cache_key = key
        return pos_embed

    def forward(self, y, x_cond, t):
        class_labels = y
        query_tokens = x_cond
        timesteps = t

        differentiable_query = query_tokens
        if self.refine is not None and not differentiable_query.requires_grad:
            differentiable_query = (
                query_tokens.clone().detach().requires_grad_(True)
            )
        _, num_tokens, _ = differentiable_query.shape

        class_embedding = self.label_emb(class_labels.reshape(-1).long())
        time_embedding = timestep_embedding(
            timesteps.reshape(-1) * 1000, self.time_dim
        )
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )

        hidden_tokens = (
            self.x_proj(differentiable_query)
            + self._position_embedding(
                num_tokens,
                differentiable_query.device,
                differentiable_query.dtype,
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden_tokens = block(hidden_tokens)
        tracked_vector = self.out_proj(hidden_tokens)
        if self.refine is None:
            return tracked_vector

        summed_potential = self.refine(tracked_vector).sum()
        return torch.autograd.grad(
            outputs=summed_potential,
            inputs=differentiable_query,
            create_graph=True,
            retain_graph=True,
        )[0]
