from functools import partial

from .mae import MAERepresentor
from .resnet import ResNetRepresentor
from .siglip2 import SigLIP2Representor


_REPRESENTOR_FACTORIES = {
    "mae_b": partial(MAERepresentor, arch="mae_b"),
    "resnet18": partial(ResNetRepresentor, arch="resnet18"),
    "siglip2_b": partial(SigLIP2Representor, arch="siglip2_b"),
}


def create_representor(arch, **kwargs):
    try:
        factory = _REPRESENTOR_FACTORIES[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown representor architecture: {arch!r}") from exc
    return factory(**kwargs)


__all__ = [
    "MAERepresentor",
    "ResNetRepresentor",
    "SigLIP2Representor",
    "create_representor",
]
