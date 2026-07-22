import torch.nn as nn

from .dit import DiT_models


class DiTBackboneAdapter(nn.Module):
    """Expose the DiT implementation through TBSM's ``(x, y, t)`` API."""

    def __init__(self, model_name="DiT-XL/2", **kwargs):
        super().__init__()
        self.model = DiT_models[model_name](**kwargs)

    def forward(self, x, y, t=None):
        output = self.model(x, t=t, c=(y,))
        return output[-1] if isinstance(output, tuple) else output


def dit_xl_2(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/2", num_classes=num_classes, **kwargs)


def dit_xl_4(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/4", num_classes=num_classes, **kwargs)
