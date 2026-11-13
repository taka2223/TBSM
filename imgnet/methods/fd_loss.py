"""Single-representation Frechet-distance post-training."""
import copy

import torch
import torch.nn as nn


def reference_sqrt(covariance):
    covariance = covariance.double()
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)) @ vectors.T


def frechet_distance(mean, covariance, reference_mean, reference_covariance,
                     reference_covariance_sqrt):
    """Differentiable Gaussian FD using a symmetric covariance product."""
    mean = mean.double()
    covariance = covariance.double()
    centered = reference_covariance_sqrt @ covariance @ reference_covariance_sqrt
    centered = 0.5 * (centered + centered.T)
    eigenvalues = torch.linalg.eigvalsh(centered).clamp_min(0)
    mean_term = (mean - reference_mean.double()).square().sum()
    covariance_term = (
        covariance.diagonal().sum()
        + reference_covariance.double().diagonal().sum()
        - 2.0 * eigenvalues.sqrt().sum()
    )
    return (mean_term + covariance_term).float()


class EMAMoments(nn.Module):
    """Population moments initialized by streaming samples, then EMA-updated."""
    def __init__(self, dimension, beta):
        super().__init__()
        self.beta = float(beta)
        if not 0.0 < self.beta < 1.0:
            raise ValueError("FD EMA beta must lie strictly between zero and one")
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("second", torch.zeros(dimension, dimension, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))

    @property
    def initialized(self):
        return bool(self.count.item() > 0)

    @torch.no_grad()
    def reset(self):
        self.mean.zero_(); self.second.zero_(); self.count.zero_()

    @torch.no_grad()
    def accumulate_initial(self, features):
        features = features.detach().double()
        self.mean.add_(features.sum(0))
        self.second.add_(features.T @ features)
        self.count.add_(features.shape[0])

    @torch.no_grad()
    def finalize_initial(self):
        count = int(self.count.item())
        if count < 2:
            raise ValueError("FD population initialization needs at least two samples")
        self.mean.div_(count)
        self.second.div_(count)

    def candidate(self, features):
        if not self.initialized:
            raise RuntimeError("FD population moments have not been initialized")
        features = features.double()
        batch_mean = features.mean(0)
        batch_second = (features.T @ features) / features.shape[0]
        mean = self.beta * self.mean.detach() + (1.0 - self.beta) * batch_mean
        second = self.beta * self.second.detach() + (1.0 - self.beta) * batch_second
        covariance = second - torch.outer(mean, mean)
        return mean, 0.5 * (covariance + covariance.T)

    @torch.no_grad()
    def update(self, features):
        features = features.detach().double()
        self.mean.mul_(self.beta).add_(features.mean(0), alpha=1.0 - self.beta)
        self.second.mul_(self.beta).addmm_(
            features.T, features, beta=1.0,
            alpha=(1.0 - self.beta) / features.shape[0],
        )
        self.count.add_(features.shape[0])


class FDLoss(nn.Module):
    """One-step SiT post-training with a frozen SigLIP Frechet objective."""
    def __init__(self, generator, representor, reference_mean, reference_covariance,
                 gen_lr, ema_decay, population_beta=0.999, fid_norm_eps=0.01,
                 noise_scale=1.0, gen_betas=(0.9, 0.999), grad_accum_steps=1):
        super().__init__()
        if int(grad_accum_steps) != 1:
            raise ValueError("FD loss does not support gradient accumulation as batch equivalence")
        self.generator = generator
        self.representor = representor.requires_grad_(False).eval()
        self.representation_fields = nn.ModuleList()
        self.noise_scale = float(noise_scale)
        self.ema_decay = float(ema_decay)
        self.fid_norm_eps = float(fid_norm_eps)
        self.grad_accum_steps = 1
        self._accum_count = 0
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in generator.backbone.parameters() if p.requires_grad],
            lr=float(gen_lr), betas=tuple(gen_betas),
        )
        self.tracker_optimizer = None
        dimension = int(reference_mean.numel())
        self.population = EMAMoments(dimension, population_beta)
        self.register_buffer("reference_mean", reference_mean.double())
        self.register_buffer("reference_covariance", reference_covariance.double())
        self.register_buffer(
            "reference_covariance_sqrt", reference_sqrt(reference_covariance)
        )
        self._eval_generator = copy.deepcopy(generator).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.representor.eval()
        return self

    def _features(self, images):
        return self.representor.extract_featmap(images).mean(dim=(2, 3))

    @torch.no_grad()
    def _ema_update(self):
        for ema_param, param in zip(
            self._eval_generator.parameters(), self.generator.parameters(), strict=True
        ):
            if param.requires_grad:
                ema_param.mul_(self.ema_decay).add_(
                    param.detach(), alpha=1.0 - self.ema_decay
                )

    @torch.no_grad()
    def initialize_population(self, count, batch_size, seed, progress=None):
        self.population.reset()
        self.generator.eval()
        device = next(self.generator.parameters()).device
        rng = torch.Generator(device=device).manual_seed(int(seed))
        latent_shape = (4, 32, 32)
        for start in range(0, int(count), int(batch_size)):
            n = min(int(batch_size), int(count) - start)
            noise = torch.randn(n, *latent_shape, device=device, generator=rng)
            noise.mul_(self.noise_scale)
            labels = torch.arange(start, start + n, device=device) % 1000
            time = torch.ones(n, device=device, dtype=noise.dtype)
            images = self.generator.to_image(self.generator(noise, labels, time))
            self.population.accumulate_initial(self._features(images))
            if progress is not None:
                progress(start + n, int(count))
        self.population.finalize_initial()
        self.generator.train()

    def step(self, real_x, real_y):
        if not self.population.initialized:
            raise RuntimeError("Initialize FD population before training")
        self.generator.train()
        noise = torch.randn_like(real_x) * self.noise_scale
        time = torch.ones(real_x.shape[0], device=real_x.device, dtype=real_x.dtype)
        images = self.generator.to_image(self.generator(noise, real_y, time))
        features = self._features(images)
        with torch.autocast(features.device.type, enabled=False):
            mean, covariance = self.population.candidate(features.float())
            raw_fd = frechet_distance(
                mean, covariance, self.reference_mean, self.reference_covariance,
                self.reference_covariance_sqrt,
            )
            loss = raw_fd / (raw_fd.detach() + self.fid_norm_eps)
        loss.backward()
        self.generator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.population.update(features)
        self._accum_count += 1
        self._ema_update()
        return {"fd_siglip": raw_fd.detach(), "generator": loss.detach()}

    @torch.no_grad()
    def sample(self, y, noise):
        self._eval_generator.eval()
        device = next(self._eval_generator.parameters()).device
        noise = noise.to(device) * self.noise_scale
        time = torch.ones(noise.shape[0], device=device, dtype=noise.dtype)
        output = self._eval_generator(noise, y.to(device), time)
        return self._eval_generator.to_image(output).float().clamp(0, 1)

    def reset_ema(self):
        self._eval_generator = copy.deepcopy(self.generator).requires_grad_(False).eval()
