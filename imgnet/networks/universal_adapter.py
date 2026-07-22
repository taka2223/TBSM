import torch
import torch.nn as nn


PREDICTION_ALIASES = {
    "x": "x",
    "x0": "x",
    "data": "x",
    "noise": "noise",
    "epsilon": "noise",
    "eps": "noise",
    "e": "noise",
    "velocity": "velocity",
    "v": "velocity",
}

INTERPOLATION_DIRECTIONS = {"data_to_noise", "noise_to_data"}


def _canonical_prediction_type(value):
    key = str(value).lower()
    if key not in PREDICTION_ALIASES:
        raise ValueError(f"Unknown prediction type: {value}")
    return PREDICTION_ALIASES[key]


def _canonical_direction(value):
    key = str(value).lower()
    if key not in INTERPOLATION_DIRECTIONS:
        raise ValueError(f"Unknown interpolation direction: {value}")
    return key


class UniversalAdapter(nn.Module):
    """Adapt a backbone between linear-flow time and prediction conventions.

    The caller uses the target convention. The wrapped backbone is called with
    the corresponding source timestep, and its source prediction is converted
    back to the target prediction type.
    """

    def __init__(
        self,
        backbone,
        *,
        source_prediction_type,
        source_interpolation_direction,
        target_prediction_type="x",
        target_interpolation_direction="data_to_noise",
    ):
        super().__init__()
        self.backbone = backbone
        self.source_prediction_type = _canonical_prediction_type(source_prediction_type)
        self.target_prediction_type = _canonical_prediction_type(target_prediction_type)
        self.source_interpolation_direction = _canonical_direction(source_interpolation_direction)
        self.target_interpolation_direction = _canonical_direction(target_interpolation_direction)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "backbone":
                raise
            backbone = self._modules.get("backbone")
            if backbone is not None and hasattr(backbone, name):
                return getattr(backbone, name)
            raise

    @staticmethod
    def _view_t(t, x):
        return t.view(t.shape[0], *([1] * (x.ndim - 1)))

    def _source_t(self, t):
        if self.source_interpolation_direction == self.target_interpolation_direction:
            return t
        return 1.0 - t

    def _x_noise_from_source(self, x_t, source_t, raw):
        t = self._view_t(source_t, x_t).to(device=x_t.device, dtype=x_t.dtype)
        raw = raw.to(dtype=x_t.dtype)
        eps = torch.finfo(x_t.dtype).eps

        if self.source_prediction_type == "x":
            x0 = raw
            if self.source_interpolation_direction == "data_to_noise":
                noise = (x_t - (1.0 - t) * x0) / t.clamp_min(eps)
            else:
                noise = (x_t - t * x0) / (1.0 - t).clamp_min(eps)
            return x0, noise

        if self.source_prediction_type == "noise":
            noise = raw
            if self.source_interpolation_direction == "data_to_noise":
                x0 = (x_t - t * noise) / (1.0 - t).clamp_min(eps)
            else:
                x0 = (x_t - (1.0 - t) * noise) / t.clamp_min(eps)
            return x0, noise

        if self.source_prediction_type == "velocity":
            velocity = raw
            if self.source_interpolation_direction == "data_to_noise":
                x0 = x_t - t * velocity
                noise = x_t + (1.0 - t) * velocity
            else:
                x0 = x_t + (1.0 - t) * velocity
                noise = x_t - t * velocity
            return x0, noise

        raise ValueError(f"Unknown source prediction type: {self.source_prediction_type}")

    def _target_from_x_noise(self, x0, noise):
        if self.target_prediction_type == "x":
            return x0
        if self.target_prediction_type == "noise":
            return noise
        if self.target_prediction_type == "velocity":
            if self.target_interpolation_direction == "data_to_noise":
                return noise - x0
            return x0 - noise
        raise ValueError(f"Unknown target prediction type: {self.target_prediction_type}")

    def forward(self, x, y, t=None, *args, **kwargs):
        if t is None:
            t = x.new_zeros(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype)
        source_t = self._source_t(t)
        raw = self.backbone(x, y, t=source_t, *args, **kwargs)
        if self.source_prediction_type == self.target_prediction_type:
            if self.source_prediction_type in {"x", "noise"}:
                return raw
            if self.source_interpolation_direction == self.target_interpolation_direction:
                return raw
            return -raw
        x0, noise = self._x_noise_from_source(x, source_t, raw)
        return self._target_from_x_noise(x0, noise)
