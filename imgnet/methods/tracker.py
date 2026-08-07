"""Conditional scattering tracker used by each representation field."""

import math

import torch
import torch.nn as nn


OUTPUT_MODES = {"field", "potential_grad", "potential"}


class ParameterFreeRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms


def timestep_embedding(timesteps, dim, max_period=10000):
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
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = ParameterFreeRMSNorm()
        self.qkv = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        )
        self.proj = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.norm2 = ParameterFreeRMSNorm()
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim, mlp_dim, bias=False)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(mlp_dim, hidden_dim, bias=False)),
        )

    def forward(self, x):
        h = self.norm1(x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(h)
        x = x + self.mlp(self.norm2(x))
        return x


class ScatteringTracker(nn.Module):
    """Return a tracked field, a potential gradient field, or a potential."""

    def __init__(
        self,
        num_classes,
        feat_dim,
        hidden_dim=736,
        time_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        output_mode="field",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        if output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"Unknown tracker output_mode: {output_mode}; "
                f"expected one of {sorted(OUTPUT_MODES)}"
            )
        self.output_mode = output_mode
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
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.potential_head = (
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1, bias=False)),
            )
            if output_mode != "field"
            else None
        )

    def _position_embedding(self, num_tokens, device, dtype):
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
            pos_embed = torch.cat(
                [
                    _sincos_embedding_1d(pos_y, dim_y),
                    _sincos_embedding_1d(pos_x, dim_x),
                ],
                dim=-1,
            )
        else:
            pos = torch.arange(num_tokens, device=device)
            pos_embed = _sincos_embedding_1d(pos, self.hidden_dim)

        pos_embed = pos_embed.to(dtype=dtype).unsqueeze(0)
        self._pos_embed_cache = pos_embed
        self._pos_embed_cache_key = key
        return pos_embed

    def forward(self, condition, query, time):
        class_labels = condition
        query_tokens = query
        timesteps = time
        differentiable_query = query_tokens
        if self.output_mode == "potential_grad" and not query_tokens.requires_grad:
            differentiable_query = query_tokens.clone().detach().requires_grad_(True)
        _, num_tokens, _ = differentiable_query.shape

        class_embedding = self.label_emb(class_labels.reshape(-1).long())
        time_embedding = timestep_embedding(timesteps.reshape(-1) * 1000, self.time_dim)
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )

        hidden_tokens = (
            self.x_proj(differentiable_query)
            + self._position_embedding(
                num_tokens, differentiable_query.device, differentiable_query.dtype
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden_tokens = block(hidden_tokens)
        predicted_field = self.out_proj(hidden_tokens)
        if self.output_mode == "field":
            return predicted_field

        token_potential = self.potential_head(predicted_field)
        if self.output_mode == "potential":
            return token_potential.mean(dim=1).squeeze(-1)
        return torch.autograd.grad(
            outputs=token_potential.sum(),
            inputs=differentiable_query,
            create_graph=True,
            retain_graph=True,
        )[0]
