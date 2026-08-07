import torch.nn as nn

from .model import DiT_models


class DiTBackboneAdapter(nn.Module):
    """Expose DiT through the common ``(x, y, t)`` backbone API."""

    def __init__(self, model_name="DiT-XL/2", **kwargs):
        super().__init__()
        self.model = DiT_models[model_name](**kwargs)

    def forward(self, x, y, t=None):
        return self.model(x, t=t, c=(y,))


def dit_xl_2(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/2", num_classes=num_classes, **kwargs)


def dit_xl_4(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/4", num_classes=num_classes, **kwargs)


def dit_b_2(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-B/2", num_classes=num_classes, **kwargs)
