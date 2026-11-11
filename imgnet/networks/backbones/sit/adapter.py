import torch.nn as nn

from .model import SiT_models


class SiTBackboneAdapter(nn.Module):
    """Official SiT weights and native noise-to-data velocity convention."""

    def __init__(self, model_name="SiT-L/2", **kwargs):
        super().__init__()
        self.model = SiT_models[model_name](**kwargs)
        # Keep the 1001-row pretrained embedding but never drop the condition:
        # sources and projectiles must belong to the same conditional law.
        self.model.y_embedder.dropout_prob = 0.0

    def forward(self, x, y, t=None):
        return self.model(x, t, y)


def sit_l_2(num_classes=1000, **kwargs):
    return SiTBackboneAdapter("SiT-L/2", num_classes=num_classes, **kwargs)
