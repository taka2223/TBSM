"""Analytic-frequency sliced Gaussian MMD field, matching the two-moons kernel.

Directions span an entire flattened image feature map, not individual tokens.
The full vector tracker marginalizes both reference and direction sampling.
"""
import math

import torch
from torch import nn


def projected_force(projectile, real, generated, directions, bandwidth, lambda_weight=1.0):
    """Sum of vector forces for explicit unit directions; caller divides by M."""
    dr = (real - projectile) @ directions.T
    ds = (generated - projectile) @ directions.T
    inv_l2 = float(bandwidth) ** -2
    fr = dr * torch.exp(-0.5 * dr.square() * inv_l2) * inv_l2
    fs = ds * torch.exp(-0.5 * ds.square() * inv_l2) * inv_l2
    return (fr - lambda_weight * fs) @ directions


class SlicedGaussianField(nn.Module):
    def __init__(self, directions=2048, bandwidth=1.0, chunk_size=128,
                 seed=20000, scale=1.0):
        super().__init__()
        if directions < 1 or chunk_size < 1 or bandwidth <= 0 or scale <= 0:
            raise ValueError("SIGReg counts, bandwidth and scale must be positive")
        self.directions, self.chunk_size = int(directions), int(chunk_size)
        self.bandwidth, self.scale = float(bandwidth), float(scale)
        self.seed = int(seed)
        self.register_buffer("draw_index", torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def forward(self, projectile, real, generated, lambda_weight=1.0):
        # Explicit FP32 projection/kernel/accumulation; no BF16 exp or reductions.
        with torch.autocast(projectile.device.type, enabled=False):
            h, r, s = [x.detach().float().flatten(1) for x in (projectile, real, generated)]
            rng = torch.Generator(device=h.device).manual_seed(self.seed + int(self.draw_index))
            self.draw_index.add_(1)
            total = torch.zeros_like(h)
            for start in range(0, self.directions, self.chunk_size):
                count = min(self.chunk_size, self.directions - start)
                a = torch.randn(count, h.shape[1], generator=rng, device=h.device)
                a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
                total.add_(projected_force(h, r, s, a, self.bandwidth, lambda_weight))
            return (total * (self.scale / self.directions)).view_as(projectile)


class IMQGaussianField(nn.Module):
    """High-dimensional approximation to the direction-averaged Gaussian field.

    For unit sphere directions and d flattened features, the averaged kernel is
    approximately (1 + ||r-h||^2 / (d * bandwidth^2))^(-1/2). Its query gradient
    is used here, with the same attraction-minus-repulsion convention and fixed
    scale as SlicedGaussianField. This is not the finite-d hypergeometric kernel;
    in particular, it is not an exact replacement for slicing on two moons.

    No directions, RNG state, pairwise batch matrix, or new trainable parameters
    are needed. Each query still uses one real and one generated reference.
    """

    def __init__(self, bandwidth=1.0, scale=1.0):
        super().__init__()
        self.bandwidth, self.scale = float(bandwidth), float(scale)
        if not all(math.isfinite(v) and v > 0 for v in (self.bandwidth, self.scale)):
            raise ValueError("IMQ bandwidth and scale must be finite and positive")

    @torch.no_grad()
    def forward(self, projectile, real, generated, lambda_weight=1.0):
        if (
            projectile.ndim < 2
            or real.shape != projectile.shape
            or generated.shape != projectile.shape
        ):
            raise ValueError("IMQ inputs must share a [batch, ...features] shape")
        # Keep reductions and kernel coefficients in FP32 under BF16 autocast.
        with torch.autocast(projectile.device.type, enabled=False):
            h, r, s = [x.detach().float().flatten(1) for x in (projectile, real, generated)]
            if h.shape[1] < 2:
                raise ValueError("Direction-averaged IMQ requires at least two features")
            inv_d_l2 = 1.0 / (h.shape[1] * self.bandwidth ** 2)

            def force(reference):
                offset = reference - h
                q = offset.square().sum(dim=1, keepdim=True) * inv_d_l2
                return offset * ((1.0 + q).pow(-1.5) * inv_d_l2)

            return ((force(r) - lambda_weight * force(s)) * self.scale).view_as(projectile)
