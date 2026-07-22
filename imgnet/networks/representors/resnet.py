import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class ResNetRepresentor(nn.Module):
    _MODEL_ALIASES = {
        "resnet18": "resnet18",
    }
    _DEFAULT_CKPTS = {
        "resnet18": pretrained_path("resnet", "resnet18-f37072fd.pth"),
    }

    def __init__(self, arch=None, input_size=None, ckpt_path=None):
        super().__init__()
        import torchvision.models as models

        weights_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        }
        model_name = self._MODEL_ALIASES.get(arch, arch)
        builder, weights, self.feat_dim = weights_map[model_name]
        ckpt_path = ckpt_path or self._DEFAULT_CKPTS.get(model_name)
        if ckpt_path is not None:
            import os

            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt_path}")
        self.featmap_dim = self.feat_dim
        if input_size is None:
            raise ValueError("ResNetRepresentor requires input_size")
        self.input_size = input_size

        backbone = builder(weights=None if ckpt_path is not None else weights)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("model", "state_dict", "module", "encoder"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            backbone.load_state_dict(state, strict=True)
        children = list(backbone.children())
        self.features = nn.Sequential(*children[:-1])
        self.features_no_avg = nn.Sequential(*children[:-2])
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def extract(self, images, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return self._forward(images)
        return self._forward(images)

    def extract_featmap(self, images):
        return self._forward_featmap(images)

    def _preprocess(self, images):
        x = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def _forward(self, images):
        return self.features(self._preprocess(images)).flatten(1)

    def _forward_featmap(self, images):
        return self.features_no_avg(self._preprocess(images))
