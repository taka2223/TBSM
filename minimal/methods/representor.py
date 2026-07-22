"""Frozen representation maps used by representation-space scattering."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityRepresentor(nn.Module):
    """Identity representation map for pixel-space scattering."""

    feat_dim = 3

    def extract_featmap(self, images):
        return images


class ResNetRepresentor(nn.Module):
    """Frozen ResNet-18 representation map used for feature-space scattering."""

    feat_dim = 512

    def __init__(self, input_size, ckpt_path=None):
        super().__init__()
        from torchvision import models

        if ckpt_path is not None and not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt_path}")
        backbone = models.resnet18(
            weights=None if ckpt_path is not None else models.ResNet18_Weights.DEFAULT
        )
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("model", "state_dict", "module", "encoder"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            backbone.load_state_dict(state, strict=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.input_size = input_size
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def extract_featmap(self, images):
        images = F.interpolate(
            images, size=self.input_size, mode="bilinear", align_corners=False
        )
        return self.features((images - self._mean) / self._std)


def build_representor(config):
    """Build the configured frozen representation map."""

    architecture = config["arch"]
    if architecture == "identity":
        return IdentityRepresentor()
    if architecture == "resnet18":
        return ResNetRepresentor(
            input_size=config["input_size"],
            ckpt_path=config.get("ckpt_path"),
        )
    raise ValueError(f"Unknown representor: {architecture}")


# Paper-aligned names. The original names remain available for API compatibility.
IdentityRepresentation = IdentityRepresentor
ResNet18Representation = ResNetRepresentor
build_representation = build_representor
