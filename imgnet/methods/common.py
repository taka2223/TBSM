import copy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class Generator(nn.Module):
    def __init__(self, backbone, decoder=None):
        super().__init__()
        self.backbone = backbone
        # The frozen decoder is generator infrastructure: share it with EMA
        # copies and keep it out of optimizer/checkpoint state.
        object.__setattr__(self, "decoder", decoder)

    def __deepcopy__(self, memo):
        copied = type(self)(copy.deepcopy(self.backbone, memo), decoder=self.decoder)
        memo[id(self)] = copied
        return copied

    def forward(self, x, y, t=None):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        return self.backbone(x, y, t=t)

    def unwrapped_forward(self, x, y, t=None):
        """Run an extra forward without touching DDP reducer or buffer sync."""
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        backbone = self.backbone
        if isinstance(backbone, DistributedDataParallel):
            backbone = backbone.module
        return backbone(x, y, t=t)

    def to_image(self, x):
        if self.decoder is not None:
            return self.decoder(x)
        return x * 0.5 + 0.5
