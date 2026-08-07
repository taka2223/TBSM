import copy
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from methods.tracker import ScatteringTracker


def _distributed_mean(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.div_(dist.get_world_size())
    return x


class RepresentationScatteringField(nn.Module):
    """Frozen-target three-body scattering regression in one feature space."""

    def __init__(
        self,
        representor,
        lambda_weight,
        rho,
        num_classes,
        tracker_config=None,
        ema_beta=0.999,
        loss_weight=1.0,
        feature_norm=("mu", "std", "rms"),
    ):
        super().__init__()
        self.representor = representor
        self.lambda_weight = float(lambda_weight)
        self.rho = float(rho)
        self.loss_weight = float(loss_weight)

        if not isinstance(feature_norm, (list, tuple)):
            raise TypeError("feature_norm must be a list or tuple")
        supported_norms = {"mu", "std", "rms"}
        unknown_norms = set(feature_norm) - supported_norms
        if unknown_norms:
            raise ValueError(
                f"Unknown feature_norm values: {sorted(unknown_norms)}; "
                f"expected a subset of {sorted(supported_norms)}"
            )
        if len(set(feature_norm)) != len(feature_norm):
            raise ValueError("feature_norm values must be unique")
        self.feature_norm = tuple(feature_norm)

        self.register_buffer("mu_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("x2_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("_ema_inited", torch.zeros(1, dtype=torch.bool))
        self.beta = ema_beta
        tracker_cfg = {} if tracker_config is None else dict(tracker_config)
        self.tracker = (
            ScatteringTracker(num_classes, feat_dim=representor.feat_dim, **tracker_cfg)
            if self.rho > 0
            else None
        )

    def _normalize_features(self, real_source, projectile, generated_source=None):
        channels = real_source.shape[-1]
        use_mu = "mu" in self.feature_norm
        use_std = "std" in self.feature_norm
        use_rms = "rms" in self.feature_norm

        if use_mu or use_std:
            mu_new = real_source.mean(dim=(0, 1)).detach().float()
            mu_new = _distributed_mean(mu_new)
            x2_new = None
            if use_std:
                x2_new = (real_source.float() ** 2).mean(dim=(0, 1)).detach()
                x2_new = _distributed_mean(x2_new)

            if not bool(self._ema_inited.item()):
                self.mu_ema.copy_(mu_new)
                if x2_new is not None:
                    self.x2_ema.copy_(x2_new)
                self._ema_inited.fill_(True)
            else:
                self.mu_ema.lerp_(mu_new, 1.0 - self.beta)
                if x2_new is not None:
                    self.x2_ema.lerp_(x2_new, 1.0 - self.beta)

        mu = self.mu_ema.view(1, 1, channels) if use_mu else None
        std = None
        if use_std:
            var = self.x2_ema - self.mu_ema.square()
            std = torch.sqrt(torch.clamp(var, min=1e-5)).view(1, 1, channels)

        def norm(x):
            if x is None:
                return None
            if mu is not None:
                x = x - mu
            if std is not None:
                x = x / std
            if use_rms:
                x = F.normalize(x, p=2, dim=-1) * math.sqrt(channels)
            return x

        return norm(real_source), norm(projectile), norm(generated_source)

    def _extract_tokens(self, image, requires_grad):
        with torch.set_grad_enabled(requires_grad):
            featmap = self.representor.extract_featmap(image)
        return featmap.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def _frozen_target(self, real_source, projectile, generated_source, tracked_vector):
        projectile_detached = projectile.detach()
        inter_offset = (real_source - projectile_detached).flatten(1)
        intra_offset = (generated_source - projectile_detached).flatten(1)

        inter_radius = torch.linalg.vector_norm(inter_offset, ord=2, dim=-1, keepdim=True)
        intra_radius = torch.linalg.vector_norm(intra_offset, ord=2, dim=-1, keepdim=True)
        radius_scale = math.sqrt(projectile.shape[-1] * projectile.shape[-2])

        b_r = inter_offset / (inter_radius + 1e-6) * radius_scale
        b_s = intra_offset / (intra_radius + 1e-6) * radius_scale
        # v_lambda = b_r - lambda * b_s.
        lambda_vector = (b_r - b_s * self.lambda_weight).view_as(projectile)
        # v_mix = (1 - rho) * v_lambda + rho * v_trk.
        mixed_vector = lambda_vector if tracked_vector is None else (
            tracked_vector * self.rho + lambda_vector * (1 - self.rho)
        )
        target = (projectile + mixed_vector).detach()
        return target, lambda_vector

    def regression_losses(
        self, real_image, projectile_image, generated_source_image, condition, time
    ):
        real_source = self._extract_tokens(real_image, requires_grad=False)
        projectile = self._extract_tokens(projectile_image, requires_grad=True)
        generated_source = None
        if self.lambda_weight > 0 and generated_source_image is not None:
            generated_source = self._extract_tokens(
                generated_source_image, requires_grad=False
            )
        real_source, projectile, generated_source = self._normalize_features(
            real_source, projectile, generated_source
        )

        if generated_source is None:
            generated_source = projectile.detach()
        # The tracker query lies on the fake-to-real corridor with alpha ~ U(0, 1-lambda).
        alpha = torch.rand(
            projectile.shape[0], 1, 1, device=projectile.device
        ) * (1 - self.lambda_weight)
        query = alpha * real_source + (1 - alpha) * projectile.detach()
        tracked_vector = (
            None
            if self.tracker is None
            else self.tracker(condition, query, time)
        )
        target, lambda_vector = self._frozen_target(
            real_source, projectile, generated_source, tracked_vector
        )
        tracked_vector = lambda_vector if tracked_vector is None else tracked_vector
        raw_tracker_loss = F.mse_loss(tracked_vector, lambda_vector)
        raw_projectile_loss = F.mse_loss(projectile, target)
        projectile_loss = (
            self.loss_weight
            * raw_projectile_loss
            / raw_projectile_loss.detach().clamp_min(1e-8)
        )
        tracker_loss = (
            self.loss_weight
            * raw_tracker_loss
            / raw_tracker_loss.detach().clamp_min(1e-8)
        )

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                projectile.mean(dim=1), real_source.mean(dim=1), dim=-1
            ).mean().item()

        loss_dict = {
            "generator": raw_projectile_loss.detach(),
            "tracker": raw_tracker_loss.detach(),
            "similarity": cos_sim,
        }
        return projectile_loss, tracker_loss, loss_dict


class TBSM(nn.Module):
    """One-step generator training with tracked three-body scattering."""

    def __init__(
        self,
        generator,
        representation_fields,
        gen_lr,
        tracker_lr,
        ema_decay,
        t_sampling,
        noise_scale=1.0,
        gen_betas=(0.9, 0.999),
        tracker_betas=(0.9, 0.999),
        grad_accum_steps=1,
    ):
        super().__init__()
        self.generator = generator
        if not isinstance(t_sampling, (list, tuple)):
            raise TypeError("t_sampling must be a list: [] for uniform or values for choice sampling")
        self.t_sampling = tuple(float(value) for value in t_sampling)
        self.noise_scale = float(noise_scale)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self._accum_count = 0
        self.representation_fields = nn.ModuleList(representation_fields or [])
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in generator.backbone.parameters() if p.requires_grad],
            lr=gen_lr,
            betas=tuple(gen_betas),
        )
        tracker_params = [
            p
            for field in self.representation_fields
            if field.tracker is not None
            for p in field.tracker.parameters()
        ]
        self.tracker_optimizer = (
            torch.optim.AdamW(
                tracker_params,
                lr=tracker_lr,
                betas=tuple(tracker_betas),
                weight_decay=0.0,
            )
            if tracker_params
            else None
        )

        self.ema_decay = float(ema_decay)
        self._eval_generator = copy.deepcopy(generator).requires_grad_(False).eval()

    @torch.no_grad()
    def _ema_update(self):
        for ema_param, param in zip(
            self._eval_generator.parameters(),
            self.generator.parameters(),
            strict=True,
        ):
            if param.requires_grad:
                ema_param.mul_(self.ema_decay).add_(
                    param.detach(), alpha=1.0 - self.ema_decay
                )

    def _sample_time(self, real_x):
        batch_size = real_x.shape[0]
        if not self.t_sampling:
            return torch.rand(
                batch_size, device=real_x.device, dtype=real_x.dtype
            )
        values = real_x.new_tensor(self.t_sampling)
        indices = torch.randint(
            values.numel(), (batch_size,), device=real_x.device
        )
        return values[indices]

    @torch.no_grad()
    def _generated_source_image(self, real_x, y):
        if not any(field.lambda_weight > 0 for field in self.representation_fields):
            return None

        self.generator.eval()
        reference = self.generator.unwrapped_forward(
            torch.randn_like(real_x) * self.noise_scale,
            y,
            real_x.new_ones(real_x.shape[0]),
        )
        generated_source_image = self.generator.to_image(reference)
        self.generator.train()
        return generated_source_image

    def _scattering_losses(
        self, real_x, projectile_x, generated_source_image, condition, time
    ):
        with torch.no_grad():
            real_image = self.generator.to_image(real_x)
        projectile_image = self.generator.to_image(projectile_x)

        generator_loss = real_x.new_tensor(0.0)
        tracker_loss = real_x.new_tensor(0.0)
        logs = {}
        multiple_fields = len(self.representation_fields) > 1
        for index, field in enumerate(self.representation_fields):
            gen_term, tracker_term, field_logs = field.regression_losses(
                real_image,
                projectile_image,
                generated_source_image,
                condition,
                time,
            )
            generator_loss = generator_loss + gen_term
            tracker_loss = tracker_loss + tracker_term
            suffix = f"_{index}" if multiple_fields else ""
            logs.update({name + suffix: value for name, value in field_logs.items()})
        return generator_loss, tracker_loss, logs

    def _update_parameters(self, generator_loss, tracker_loss):
        self._accum_count += 1
        scale = 1.0 / self.grad_accum_steps
        (generator_loss * scale).backward()
        if self.tracker_optimizer is not None:
            (tracker_loss * scale).backward()

        if self._accum_count % self.grad_accum_steps != 0:
            return

        self.generator_optimizer.step()
        if self.tracker_optimizer is not None:
            self.tracker_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        if self.tracker_optimizer is not None:
            self.tracker_optimizer.zero_grad(set_to_none=True)
        self._ema_update()

    def step(self, real_x, real_y):
        batch_size = real_x.shape[0]
        self.generator.train()

        noise = torch.randn_like(real_x) * self.noise_scale
        t = self._sample_time(real_x)
        t_img = t.view(batch_size, *([1] * (real_x.ndim - 1)))
        x_t = real_x * (1.0 - t_img) + noise * t_img
        projectile_x = self.generator(x_t, real_y, t)
        generated_source_image = self._generated_source_image(real_x, real_y)
        generator_loss, tracker_loss, logs = self._scattering_losses(
            real_x, projectile_x, generated_source_image, real_y, t
        )
        self._update_parameters(generator_loss, tracker_loss)
        return logs

    @torch.no_grad()
    def sample(self, y, noise):
        self._eval_generator.eval()
        device = next(self._eval_generator.parameters()).device
        noise = noise.to(device) * self.noise_scale
        t = torch.ones(noise.shape[0], device=device, dtype=noise.dtype)
        x = self._eval_generator(
            noise,
            y.to(device),
            t,
        )
        return self._eval_generator.to_image(x).float().clamp(0, 1)

    def reset_ema(self):
        self._eval_generator = copy.deepcopy(self.generator).requires_grad_(False).eval()
