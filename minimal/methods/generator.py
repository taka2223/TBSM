"""Minimal conditional U-Net used as the TBSM generator."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embedding used for the generator timestep condition."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        time = time * 1000.0
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class SelfAttention(nn.Module):
    """Spatial self-attention with a zero-initialized residual projection."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        hidden = self.norm(x)
        query_key_value = self.qkv(hidden).reshape(
            batch_size, 3, channels, height * width
        )
        query, key, value = query_key_value.unbind(1)
        attention = torch.bmm(query.transpose(1, 2), key) * (channels ** -0.5)
        attention = attention.softmax(dim=-1)
        hidden = torch.bmm(value, attention.transpose(1, 2))
        hidden = hidden.reshape(batch_size, channels, height, width)
        return x + self.proj(hidden)


class Block(nn.Module):
    """Conditioned residual block used throughout the generator U-Net."""

    def __init__(self, in_ch, out_ch, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_ch * 2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, cond):
        hidden = self.conv1(x)
        scale, shift = self.cond_mlp(cond).chunk(2, dim=1)
        hidden = (
            self.norm1(hidden) * (1 + scale[..., None, None])
            + shift[..., None, None]
        )
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = F.silu(hidden)
        return hidden + self.shortcut(x)


class Generator(nn.Module):
    """U-Net generator; TBSM ``t=1`` is noise and the U-Net embeds ``1-t``."""

    def __init__(self, in_ch=3, out_ch=3, cond_emb_dim=256, num_classes=1000):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(cond_emb_dim),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.SiLU(),
        )
        self.class_emb = nn.Embedding(num_classes, cond_emb_dim)

        self.b1 = Block(in_ch, 64, cond_emb_dim)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.b2 = Block(64, 128, cond_emb_dim)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.b3 = Block(128, 256, cond_emb_dim)
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        self.mid = Block(256, 256, cond_emb_dim)
        self.mid_attn = SelfAttention(256)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.b4 = Block(512, 128, cond_emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.b5 = Block(256, 64, cond_emb_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.b6 = Block(128, 64, cond_emb_dim)
        self.final_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x, y, t=None):
        model_input = x
        class_labels = y
        timesteps = t

        conditioning = self.class_emb(class_labels)
        if timesteps is not None:
            conditioning = conditioning + self.time_mlp(
                1 - timesteps.flatten()
            )

        first_skip = self.b1(model_input, conditioning)
        second_skip = self.b2(self.down1(first_skip), conditioning)
        third_skip = self.b3(self.down2(second_skip), conditioning)
        hidden = self.mid(self.down3(third_skip), conditioning)
        hidden = self.mid_attn(hidden)

        hidden = self.b4(
            torch.cat([self.up1(hidden), third_skip], dim=1), conditioning
        )
        hidden = self.b5(
            torch.cat([self.up2(hidden), second_skip], dim=1), conditioning
        )
        hidden = self.b6(
            torch.cat([self.up3(hidden), first_skip], dim=1), conditioning
        )
        return self.final_conv(hidden)

    @staticmethod
    def to_image(x):
        return x * 0.5 + 0.5
