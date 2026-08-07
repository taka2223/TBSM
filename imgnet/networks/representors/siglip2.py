from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class SigLIP2Representor(nn.Module):
    """SigLIP2 vision encoder adapted to TBSM's representor interface."""

    _MODEL_ALIASES = {
        "siglip2_b": "siglip2_b",
    }

    _MODEL_MAP = {
        "siglip2_b": "google/siglip2-base-patch16-256",
    }

    def __init__(
        self,
        arch="siglip2_b",
        input_size=256,
        ckpt_path=None,
        model_dir=None,
        use_post_layernorm=True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unsupported SigLIP2Representor kwargs: {sorted(kwargs)}")

        name = self._MODEL_ALIASES.get(arch, arch)
        if name not in self._MODEL_MAP:
            raise KeyError(
                f"Unknown SigLIP2 model: {arch}. Available options: {sorted(self._MODEL_ALIASES)}"
            )

        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:
            raise ImportError("SigLIP2Representor requires transformers.") from exc

        self.arch = name
        self.input_size = int(input_size)
        self.model_id = self._MODEL_MAP[name]
        self.use_post_layernorm = use_post_layernorm

        default_local_dir = pretrained_path("siglip2", self.model_id.split("/", 1)[1])
        local_dir = model_dir or ckpt_path or default_local_dir
        if local_dir is not None:
            local_dir = Path(local_dir)
            if local_dir.is_file():
                local_dir = local_dir.parent
            if not local_dir.is_dir():
                raise FileNotFoundError(
                    f"SigLIP2 local model directory not found: {local_dir}"
                )
            self.model = SiglipVisionModel.from_pretrained(str(local_dir))
        else:
            self.model = SiglipVisionModel.from_pretrained(self.model_id)
        self.model.eval()
        self.feat_dim = int(self.model.config.hidden_size)
        self.featmap_dim = self.feat_dim
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))

        self.register_buffer("_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

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
        x = images
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        return (x - self._mean) / self._std

    def _tokens(self, images, no_grad=True):
        if no_grad:
            with torch.no_grad():
                out = self.model(
                    self._preprocess(images),
                    output_hidden_states=not self.use_post_layernorm,
                )
        else:
            out = self.model(
                self._preprocess(images),
                output_hidden_states=not self.use_post_layernorm,
            )

        if (
            not self.use_post_layernorm
            and hasattr(out, "hidden_states")
            and out.hidden_states is not None
        ):
            out = out.hidden_states[-1]
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            out = out[-1]

        if out.ndim != 3:
            raise RuntimeError(
                f"Expected SigLIP2 tokens with shape [B, N, C], got {tuple(out.shape)}"
            )
        return out

    def _forward(self, images):
        return self._tokens(images).mean(dim=1)

    def _forward_featmap(self, images):
        tokens = self._tokens(images, no_grad=False)
        b, n, c = tokens.shape
        h = w = int(math.isqrt(n))
        if h * w != n:
            raise RuntimeError(
                f"Cannot reshape {n} SigLIP2 patch tokens into a square feature map"
            )
        return tokens.transpose(1, 2).reshape(b, c, h, w)
