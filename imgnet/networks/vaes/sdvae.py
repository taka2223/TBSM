import torch.nn as nn


class SDVAE(nn.Module):
    """Frozen Stable Diffusion VAE with scaled-latent encode/decode helpers."""

    def __init__(self, model_path, multiplier=0.18215):
        super().__init__()
        try:
            from diffusers.models import AutoencoderKL
        except ImportError as exc:
            raise ImportError("SDVAE requires diffusers.") from exc

        self.model = AutoencoderKL.from_pretrained(model_path)
        self.multiplier = float(multiplier)
        self.model.requires_grad_(False).eval()
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def encode(self, image, sample=False, generator=None):
        """Encode an image in [0, 1] to a scaled SD-VAE latent."""
        posterior = self.model.encode(image * 2.0 - 1.0).latent_dist
        latent = posterior.sample(generator=generator) if sample else posterior.mean
        return latent * self.multiplier

    def decode(self, latent):
        """Decode a scaled SD-VAE latent to an image in [0, 1]."""
        image = self.model.decode(latent / self.multiplier).sample
        return ((image + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(self, latent):
        return self.decode(latent)
