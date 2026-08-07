import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class MAERepresentor(nn.Module):
    """Frozen MAE-B encoder exposing final-layer patch-token feature maps."""

    _MODEL_ALIASES = {
        "mae_b": "vit_base_patch16_224.mae",
    }
    _DEFAULT_CKPTS = {
        "vit_base_patch16_224.mae": pretrained_path(
            "mae", "vit_base_patch16_224.mae.pth"
        ),
    }

    def __init__(self, arch, input_size=224, ckpt_path=None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "MAERepresentor requires timm to use arch='mae_b'."
            ) from exc

        if arch not in self._MODEL_ALIASES:
            raise KeyError(
                f"Unknown MAE model: {arch}. Available options: {sorted(self._MODEL_ALIASES)}"
            )
        model_name = self._MODEL_ALIASES[arch]
        ckpt_path = ckpt_path or self._DEFAULT_CKPTS.get(model_name)
        if ckpt_path is not None and not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")
        pretrained = ckpt_path is None
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if ckpt_path is not None:
            try:
                from safetensors.torch import load_file

                state = load_file(str(ckpt_path), device="cpu")
            except Exception:
                state = torch.load(Path(ckpt_path), map_location="cpu")
            if isinstance(state, dict):
                for key in ("model", "state_dict", "module", "encoder"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            self.net.load_state_dict(state, strict=True)
        self.input_size = input_size
        self.feat_dim = self.net.num_features
        self.featmap_dim = self.feat_dim
        self.num_prefix_tokens = getattr(self.net, "num_prefix_tokens", 1)
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

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
        x = F.interpolate(
            images, size=self.input_size, mode="bicubic", align_corners=False
        )
        return (x - self._mean) / self._std

    def _patch_tokens(self, images):
        tokens = self.net.forward_features(self._preprocess(images))
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected MAE ViT tokens with shape [B, N, C], got {tuple(tokens.shape)}"
            )
        return tokens[:, self.num_prefix_tokens :]

    def _forward(self, images):
        return self._patch_tokens(images).mean(dim=1)

    def _forward_featmap(self, images):
        tokens = self._patch_tokens(images)
        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise RuntimeError(
                f"Cannot reshape {N} MAE patch tokens into a square feature map"
            )
        return tokens.transpose(1, 2).reshape(B, C, H, W)
