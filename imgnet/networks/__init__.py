from .backbones import (
    dit_xl_2,
    dit_xl_4,
    jit_b,
    jit_h,
    jit_l,
    pixeldit_b,
    pixeldit_xl,
)
from .universal_adapter import UniversalAdapter


_BACKBONE_FACTORIES = {
    "dit_xl_2": dit_xl_2,
    "dit_xl_4": dit_xl_4,
    "jit_b": jit_b,
    "jit_h": jit_h,
    "jit_l": jit_l,
    "pixeldit_b": pixeldit_b,
    "pixeldit_xl": pixeldit_xl,
}


def create_backbone(name, *, num_classes, **kwargs):
    try:
        factory = _BACKBONE_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown network: {name}") from exc
    return factory(num_classes=num_classes, **kwargs)


__all__ = [
    "UniversalAdapter",
    "create_backbone",
    "dit_xl_2",
    "dit_xl_4",
    "jit_b",
    "jit_h",
    "jit_l",
    "pixeldit_b",
    "pixeldit_xl",
]
