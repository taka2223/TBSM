import copy
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from methods.tracker import Tracker


SCATTERING_EPS = 1e-6

def _self_normalized(loss, weight):
    return weight * loss / loss.detach().clamp_min(1e-8)


def _shadow(module):
    return {name: param.clone().detach() for name, param in module.named_parameters()}


def _freeze(module):
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


def _distributed_mean(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.div_(dist.get_world_size())
    return x


class RepresentationScattering(nn.Module):
    """TBSM scattering objective in one frozen representation space."""
    def __init__(
        self,
        representor,
        intra_weight,  # intra-source weight lambda
        track_weight,  # tracked-supervision weight rho
        tracker=None,
        ema_beta=0.999,
        loss_weight=1.0,
    ):
        super().__init__()
        self.representor = representor
        # Lambda controls the intra-source coefficient in
        # hat_v_lambda = b_r - lambda * b_s and the tracker-query range
        # alpha ~ U(0, 1-lambda). Lambda=1 is the energy-distance endpoint.
        self.lam = float(intra_weight)
        # Rho mixes the instantaneous scattering vector and tracked vector:
        # hat_v_mix = (1-rho) * hat_v_lambda + rho * v_trk.
        self.rho = float(track_weight)
        self.loss_weight = float(loss_weight)

        self.tracker_cfg = {} if tracker is None else dict(tracker)
        self.tracker = None

        self.register_buffer("mu_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("x2_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("_ema_inited", torch.zeros(1, dtype=torch.bool))
        self.beta = ema_beta

    @property
    def representation_map(self):
        return self.representor

    @property
    def intra_source_weight(self):
        return self.lam

    @property
    def tracked_supervision_weight(self):
        return self.rho

    def init_tracker(self, num_classes):
        if self.rho <= 0 or self.tracker is not None:
            return
        self.tracker = Tracker(num_classes, feat_dim=self.mu_ema.numel(), **self.tracker_cfg)
        self.tracker.to(device=self.mu_ema.device)

    def _normalize_bodies(self, x_r, x_p, x_s=None):
        channels = x_r.shape[-1]
        mu_new = x_r.mean(dim=(0, 1)).detach().float()
        x2_new = (x_r ** 2).mean(dim=(0, 1)).detach().float()
        mu_new = _distributed_mean(mu_new)
        x2_new = _distributed_mean(x2_new)
        if not bool(self._ema_inited.item()):
            self.mu_ema.copy_(mu_new)
            self.x2_ema.copy_(x2_new)
            self._ema_inited.fill_(True)
        else:
            self.mu_ema.lerp_(mu_new, 1.0 - self.beta)
            self.x2_ema.lerp_(x2_new, 1.0 - self.beta)

        var = self.x2_ema - (self.mu_ema ** 2)
        mu = self.mu_ema.view(1, 1, channels)
        std = torch.sqrt(torch.clamp(var, min=1e-5)).view(1, 1, channels)

        def norm(x):
            if x is None:
                return None
            x = (x - mu) / std
            return F.normalize(x, p=2, dim=-1) * math.sqrt(channels)

        return norm(x_r), norm(x_p), norm(x_s)

    def _feature_tokens(self, image, requires_grad):
        with torch.set_grad_enabled(requires_grad):
            featmap = self.representor.extract_featmap(image)
        b, c, h, w = featmap.shape
        return featmap.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def feature_tokens(self, image, requires_grad=True):
        """Extract spatial tokens from the representation model's final feature map."""
        return self._feature_tokens(image, requires_grad=requires_grad)

    def _resolve_fake_source(self, x_p, x_s):
        if self.lam <= 0:
            x_s = None
        return x_p.detach() if x_s is None else x_s

    def _tracked_vector(self, y, tracker_condition, t):
        return None if self.tracker is None else self.tracker(y, tracker_condition, t)

    @torch.no_grad()
    def compute_three_body_scattering_target(self, x_r, x_p, x_s, v_trk=None):
        """Form a detached mixed-scattering target from one representation-space triplet."""
        x_p_det = x_p.detach()
        inter_offset = (x_r - x_p_det).flatten(1)  # projectile-to-real-source offset
        intra_offset = (x_s - x_p_det).flatten(1)  # projectile-to-generated-source offset

        inter_radius = torch.linalg.vector_norm(inter_offset, ord=2, dim=-1, keepdim=True)
        intra_radius = torch.linalg.vector_norm(intra_offset, ord=2, dim=-1, keepdim=True)
        radius_scale = math.sqrt(x_p.shape[-1] * x_p.shape[-2])

        # RMS-scaled inter- and intra-source bearings b_r and b_s.
        b_r = inter_offset / (inter_radius + SCATTERING_EPS) * radius_scale
        b_s = intra_offset / (intra_radius + SCATTERING_EPS) * radius_scale
        # Lambda-weighted scattering vector hat_v_lambda = b_r - lambda * b_s.
        v_lambda = (b_r - b_s * self.lam).view_as(x_p)
        # Rho-mixed scattering vector hat_v_mix = (1-rho) * hat_v_lambda + rho * v_trk.
        v_mix = v_lambda if v_trk is None else (
            v_trk * self.rho + v_lambda * (1 - self.rho)
        )
        # Detached generator target x_target = sg[x_p + hat_v_mix].
        x_target = (x_p + v_mix).detach()
        return x_target, v_lambda

    def compute_loss(self, real_img, fake_img, fref_img, y, t):
        x_r = self.feature_tokens(real_img, requires_grad=False)
        x_p = self.feature_tokens(fake_img, requires_grad=True)
        x_s = None
        if self.lam > 0 and fref_img is not None:
            x_s = self.feature_tokens(fref_img, requires_grad=False)
        x_r, x_p, x_s = self._normalize_bodies(x_r, x_p, x_s)

        batch = x_p.shape[0]
        x_s = self._resolve_fake_source(x_p, x_s)
        # Query the tracker at x_tilde = alpha*x_r + (1-alpha)*sg[x_p],
        # where alpha ~ U(0, 1-lambda).
        alpha = torch.rand(batch, 1, 1, device=x_p.device) * (1 - self.lam)
        x_tilde = alpha * x_r + (1 - alpha) * x_p.detach()
        v_trk = self._tracked_vector(y, x_tilde, t)
        x_target, v_lambda = self.compute_three_body_scattering_target(
            x_r, x_p, x_s, v_trk
        )
        v_trk = v_trk if v_trk is not None else v_lambda
        raw_tracker_loss = F.mse_loss(v_trk, v_lambda)
        raw_projectile_loss = F.mse_loss(x_p, x_target)
        projectile_loss = _self_normalized(raw_projectile_loss, self.loss_weight)
        tracker_loss = _self_normalized(raw_tracker_loss, self.loss_weight)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                x_p.mean(dim=1), x_r.mean(dim=1), dim=-1
            ).mean().item()

        loss_dict = {
            "generator": raw_projectile_loss.detach(),
            "tracker": raw_tracker_loss.detach(),
            "similarity": cos_sim,
        }
        return projectile_loss, tracker_loss, loss_dict


class TBSM(nn.Module):
    """Train a one-step generator with tracked three-body scattering regression."""
    def __init__(
        self,
        generator,
        matchers,
        num_classes,
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
        self.gen_betas = tuple(gen_betas)
        self.tracker_betas = tuple(tracker_betas)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self._accum_count = 0
        self.matchers = nn.ModuleList(list(matchers) if matchers is not None else [])
        for matcher in self.matchers:
            matcher.init_tracker(num_classes)
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in generator.backbone.parameters() if p.requires_grad], lr=gen_lr, betas=self.gen_betas
        )
        tracker_params = [
            p
            for matcher in self.matchers
            if matcher.tracker is not None
            for p in matcher.tracker.parameters()
        ]
        for p in tracker_params:
            p.requires_grad_(True)
        self.tracker_optimizer = (
            torch.optim.AdamW(tracker_params, lr=tracker_lr, betas=self.tracker_betas, weight_decay=0.0)
            if tracker_params
            else None
        )

        self.ema_decay = ema_decay
        self._ema_shadow = _shadow(generator)
        self._eval_generator = _freeze(copy.deepcopy(generator))
        self._ema_initialized = False

    @property
    def representation_fields(self):
        """Return the representation-specific scattering objectives."""
        return self.matchers

    def _ensure_ema_device(self):
        if not self._ema_initialized:
            device = next(self.generator.parameters()).device
            self._ema_shadow = {n: v.to(device) for n, v in self._ema_shadow.items()}
            self._ema_initialized = True

    def _ema_update(self):
        self._ensure_ema_device()
        with torch.no_grad():
            for name, param in self.generator.named_parameters():
                key = name.replace("backbone.module.", "backbone.")
                if key in self._ema_shadow and param.requires_grad:
                    self._ema_shadow[key].mul_(self.ema_decay).add_(
                        param.detach(), alpha=1.0 - self.ema_decay)

    def _apply_ema(self):
        self._ensure_ema_device()
        for name, param in self._eval_generator.named_parameters():
            if name in self._ema_shadow:
                param.data.copy_(self._ema_shadow[name])

    def state_dict(self, *args, **kwargs):
        self._apply_ema()
        return super().state_dict(*args, **kwargs)

    def _generate_from_noise(self, generator, noise, y, t=None):
        if t is None:
            t = torch.ones(noise.shape[0], device=noise.device, dtype=noise.dtype)
        return generator(noise, y, t)

    def step(self, real_x, real_y):
        batch_size = real_x.shape[0]
        device = real_x.device
        self.generator.train()

        noise = torch.randn_like(real_x) * self.noise_scale
        if len(self.t_sampling) == 0:
            t = torch.rand(batch_size, device=device, dtype=real_x.dtype)
        else:
            values = real_x.new_tensor(self.t_sampling)
            t = values[torch.randint(values.numel(), (batch_size,), device=device)]

        t_img = t.view(batch_size, *([1] * (real_x.ndim - 1)))
        # Form the configured paired noisy input; t=1 gives pure noise.
        x_t = real_x * (1.0 - t_img) + noise * t_img

        y = real_y.clone()
        generated_x = self.generator(x_t, y, t)

        fref_img = None
        if any(matcher.lam > 0 for matcher in self.matchers):
            with torch.no_grad():
                self.generator.eval()
                # The generated source uses independent pure noise at t=1,
                # regardless of the projectile timestep.
                fref_x = self.generator.auxiliary_forward(
                    torch.randn_like(real_x) * self.noise_scale,
                    y,
                    real_x.new_ones(real_x.shape[0]),
                )
                fref_img = self.generator.to_image(fref_x)
                self.generator.train()

        with torch.no_grad():
            real_img = self.generator.to_image(real_x)
        fake_img = self.generator.to_image(generated_x)

        total_gen_loss = real_x.new_tensor(0.0)
        total_tracker_loss = real_x.new_tensor(0.0)
        loss_dict = {}
        for index, matcher in enumerate(self.matchers):
            gen_loss, tracker_loss, matcher_log = matcher.compute_loss(
                real_img, fake_img, fref_img, y, t
            )
            total_gen_loss = total_gen_loss + gen_loss
            if tracker_loss.requires_grad:
                total_tracker_loss = total_tracker_loss + tracker_loss
            suffix = f"_{index}" if len(self.matchers) > 1 else ""
            loss_dict.update({key + suffix: value for key, value in matcher_log.items()})

        # Average gradients over microbatches; update optimizers and generator
        # EMA at each accumulation boundary.
        self._accum_count += 1
        accum_scale = 1.0 / self.grad_accum_steps
        (total_gen_loss * accum_scale).backward(create_graph=False)
        if self.tracker_optimizer is not None and total_tracker_loss.requires_grad:
            (total_tracker_loss * accum_scale).backward(create_graph=False)
        if self._accum_count % self.grad_accum_steps == 0:
            self.generator_optimizer.step()
            if self.tracker_optimizer is not None and total_tracker_loss.requires_grad:
                self.tracker_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            if self.tracker_optimizer is not None:
                self.tracker_optimizer.zero_grad(set_to_none=True)
            self._ema_update()
        return loss_dict

    @torch.no_grad()
    def sample(self, y, noise):
        self._apply_ema()
        self._eval_generator.eval()
        device = next(self._eval_generator.parameters()).device
        x = self._generate_from_noise(
            self._eval_generator, noise.to(device) * self.noise_scale, y.to(device)
        )
        return self._eval_generator.to_image(x).float().clamp(0, 1)

    def reset_ema(self):
        self._ema_shadow = _shadow(self.generator)
        self._eval_generator = _freeze(copy.deepcopy(self.generator))
        self._ema_initialized = False

    def sync_ema_shadow(self):
        self._ema_shadow = _shadow(self._eval_generator)
        self._ema_initialized = False
