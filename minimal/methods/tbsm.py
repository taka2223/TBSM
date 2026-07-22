"""TBSM scattering objectives and one-step training orchestration."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tracker import Tracker


SCATTERING_EPS = 1e-6


def self_normalized(loss):
    """Rescale a loss by its detached value while preserving its gradient path."""

    return loss / loss.detach().clamp_min(1e-8)


class DistributionMatcher(nn.Module):
    """One representation-space TBSM scattering objective.

    The registered attribute names retain compatibility with existing minimal-code
    checkpoints. Paper-aligned aliases expose the representation map, lambda, and
    rho terminology without changing those checkpoint keys.
    """

    def __init__(
        self,
        representor,
        num_classes,
        intra_weight=1.0,
        track_weight=0.0,
        tracker=None,
        statistics_decay=0.999,
    ):
        super().__init__()
        representation_map = representor
        self.representor = representation_map
        self.lam = float(intra_weight)  # lambda: intra-source weight
        self.rho = float(track_weight)  # rho: tracked-supervision weight
        self.statistics_decay = float(statistics_decay)
        tracker_config = {} if tracker is None else dict(tracker)
        self.tracker = (
            Tracker(
                num_classes,
                feat_dim=representation_map.feat_dim,
                **tracker_config,
            )
            if self.rho > 0
            else None
        )
        self.register_buffer("mean", torch.zeros(representation_map.feat_dim))
        self.register_buffer(
            "second_moment", torch.zeros(representation_map.feat_dim)
        )
        self.register_buffer("statistics_initialized", torch.tensor(False))

    @property
    def representation_map(self):
        """Frozen representation map used by this scattering objective."""

        return self.representor

    @property
    def intra_source_weight(self):
        """Paper parameter lambda, the intra-source bearing weight."""

        return self.lam

    @property
    def tracked_supervision_weight(self):
        """Paper parameter rho, the tracked-supervision mixture weight."""

        return self.rho

    def feature_tokens(self, images):
        representation_feature_map = self.representor.extract_featmap(images)
        return representation_feature_map.permute(0, 2, 3, 1).flatten(1, 2)

    def normalize(self, real, fake, reference):
        """Update statistics, then normalize source and projectile feature tokens.

        ``real``, ``fake``, and ``reference`` are retained in the public signature
        for compatibility. They correspond to the paper's real source, projectile,
        and independently generated source, respectively.
        """

        real_source_features = real
        projectile_features = fake
        generated_source_features = reference
        with torch.no_grad():
            new_mean = real_source_features.mean((0, 1)).float()
            new_second_moment = real_source_features.square().mean((0, 1)).float()
            if not self.statistics_initialized:
                self.mean.copy_(new_mean)
                self.second_moment.copy_(new_second_moment)
                self.statistics_initialized.fill_(True)
            else:
                amount = 1 - self.statistics_decay
                self.mean.lerp_(new_mean, amount)
                self.second_moment.lerp_(new_second_moment, amount)

        std = (self.second_moment - self.mean.square()).clamp_min(1e-5).sqrt()

        def normalize_one(x):
            if x is None:
                return None
            x = (x - self.mean[None, None]) / std[None, None]
            return F.normalize(x, dim=-1) * math.sqrt(x.shape[-1])

        return (
            normalize_one(real_source_features),
            normalize_one(projectile_features),
            normalize_one(generated_source_features),
        )

    @torch.no_grad()
    def scattering_target(self, real, fake, reference, tracked_vector=None):
        """Construct the frozen generator target and lambda-weighted vector.

        The legacy argument names map to real source, projectile, and independently
        generated source features. ``tracked_vector`` is the tracker prediction
        v_trk; the returned instantaneous vector is v_lambda.
        """

        real_source_features = real
        projectile_features = fake
        generated_source_features = reference

        detached_projectile = projectile_features.detach()
        inter_source_offset = (
            real_source_features - detached_projectile
        ).flatten(1)
        intra_source_offset = (
            generated_source_features - detached_projectile
        ).flatten(1)
        radius_scale = math.sqrt(
            projectile_features.shape[1] * projectile_features.shape[2]
        )
        inter_source_bearing = inter_source_offset / (
            torch.linalg.vector_norm(
                inter_source_offset, dim=-1, keepdim=True
            )
            + SCATTERING_EPS
        ) * radius_scale
        intra_source_bearing = intra_source_offset / (
            torch.linalg.vector_norm(
                intra_source_offset, dim=-1, keepdim=True
            )
            + SCATTERING_EPS
        ) * radius_scale
        lambda_weighted_scattering = (
            inter_source_bearing - self.lam * intra_source_bearing
        ).view_as(projectile_features)
        mixed_scattering = (
            lambda_weighted_scattering
            if tracked_vector is None
            else (1 - self.rho) * lambda_weighted_scattering
            + self.rho * tracked_vector
        )
        generator_target = (
            projectile_features + mixed_scattering
        )
        return generator_target.detach(), lambda_weighted_scattering

    def forward(self, real_images, fake_images, reference_images, labels, t):
        """Evaluate one sampled real--projectile--generated-source event."""

        real_source_images = real_images
        projectile_images = fake_images
        generated_source_images = reference_images
        class_labels = labels
        timesteps = t

        real_source_features = self.feature_tokens(real_source_images).detach()
        projectile_features = self.feature_tokens(projectile_images)
        generated_source_features = self.feature_tokens(
            generated_source_images
        ).detach()
        (
            real_source_features,
            projectile_features,
            generated_source_features,
        ) = self.normalize(
            real_source_features,
            projectile_features,
            generated_source_features,
        )

        alpha = torch.rand(
            projectile_features.shape[0],
            1,
            1,
            device=projectile_features.device,
        ) * (1 - self.lam)
        tracker_query = (
            alpha * real_source_features
            + (1 - alpha) * projectile_features.detach()
        )
        tracked_vector = (
            None
            if self.tracker is None
            else self.tracker(class_labels, tracker_query, timesteps)
        )
        generator_target, lambda_weighted_scattering = self.scattering_target(
            real_source_features,
            projectile_features,
            generated_source_features,
            tracked_vector,
        )

        raw_generator_loss = F.mse_loss(
            projectile_features, generator_target
        )
        tracker_prediction = (
            lambda_weighted_scattering
            if tracked_vector is None
            else tracked_vector
        )
        raw_tracker_loss = F.mse_loss(
            tracker_prediction, lambda_weighted_scattering
        )
        similarity = F.cosine_similarity(
            projectile_features.mean(1),
            real_source_features.mean(1),
            dim=-1,
        ).mean()
        return (
            self_normalized(raw_generator_loss),
            self_normalized(raw_tracker_loss),
            {
                "generator": raw_generator_loss.detach(),
                "tracker": raw_tracker_loss.detach(),
                "similarity": similarity.detach(),
            },
        )


class TBSM(nn.Module):
    """One-step TBSM training without generator EMA or distributed machinery."""

    def __init__(
        self,
        generator,
        matchers,
        gen_lr,
        tracker_lr,
        t_sampling,
    ):
        super().__init__()
        self.generator = generator
        self.matchers = nn.ModuleList(matchers)
        if not isinstance(t_sampling, (list, tuple)):
            raise TypeError("t_sampling must be a list")
        self.t_sampling = tuple(float(value) for value in t_sampling)
        self.generator_optimizer = torch.optim.AdamW(
            generator.parameters(), lr=gen_lr
        )
        tracker_parameters = [
            parameter
            for representation_field in self.matchers
            if representation_field.tracker is not None
            for parameter in representation_field.tracker.parameters()
        ]
        self.tracker_optimizer = (
            torch.optim.AdamW(tracker_parameters, lr=tracker_lr, weight_decay=0)
            if tracker_parameters
            else None
        )

    @property
    def representation_fields(self):
        """Paper-aligned view of the registered representation matchers."""

        return self.matchers

    def step(self, real, labels):
        real_source_images = real
        class_labels = labels
        batch_size = real_source_images.shape[0]
        if self.t_sampling:
            timestep_choices = real_source_images.new_tensor(self.t_sampling)
            timesteps = timestep_choices[
                torch.randint(
                    timestep_choices.numel(),
                    (batch_size,),
                    device=real_source_images.device,
                )
            ]
        else:
            timesteps = torch.rand(
                batch_size,
                device=real_source_images.device,
                dtype=real_source_images.dtype,
            )
        projectile_noise = torch.randn_like(real_source_images)
        projectile_inputs = (
            (1 - timesteps[:, None, None, None]) * real_source_images
            + timesteps[:, None, None, None] * projectile_noise
        )
        projectile_images = self.generator(
            projectile_inputs, class_labels, timesteps
        )

        with torch.no_grad():
            generated_source_timesteps = torch.ones(
                batch_size,
                device=real_source_images.device,
                dtype=real_source_images.dtype,
            )
            generated_source_images = self.generator(
                torch.randn_like(real_source_images),
                class_labels,
                generated_source_timesteps,
            )

        total_generator_loss = real_source_images.new_tensor(0.0)
        total_tracker_loss = real_source_images.new_tensor(0.0)
        metrics = {}
        for index, representation_field in enumerate(self.matchers):
            generator_loss, tracker_loss, field_metrics = representation_field(
                self.generator.to_image(real_source_images),
                self.generator.to_image(projectile_images),
                self.generator.to_image(generated_source_images),
                class_labels,
                timesteps,
            )
            total_generator_loss = total_generator_loss + generator_loss
            if tracker_loss.requires_grad:
                total_tracker_loss = total_tracker_loss + tracker_loss
            suffix = f"_{index}" if len(self.matchers) > 1 else ""
            metrics.update(
                {key + suffix: value for key, value in field_metrics.items()}
            )
        self.generator_optimizer.zero_grad(set_to_none=True)
        if self.tracker_optimizer is not None:
            self.tracker_optimizer.zero_grad(set_to_none=True)
        total_generator_loss.backward()
        if self.tracker_optimizer is not None and total_tracker_loss.requires_grad:
            total_tracker_loss.backward()
        self.generator_optimizer.step()
        if self.tracker_optimizer is not None:
            self.tracker_optimizer.step()
        return metrics

    @torch.no_grad()
    def sample(self, noise, labels):
        pure_noise_timesteps = torch.ones(noise.shape[0], device=noise.device)
        return self.generator.to_image(
            self.generator(noise, labels, pure_noise_timesteps)
        ).clamp(0, 1)


# Paper-aligned class name. The original name remains the canonical class object
# so imports and full-object deserialization continue to resolve it.
RepresentationScattering = DistributionMatcher
