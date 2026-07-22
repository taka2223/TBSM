"""PixelDiT backbones for pixel-space TBSM training."""

from .pixdit_core.pixeldit_c2i import PixDiT as _OfficialPixDiT


class PixDiT(_OfficialPixDiT):
    """Expose PixelDiT through TBSM's ``(x, y, t)`` generator interface."""

    def forward(self, x, y, t=None, s=None, mask=None):
        if t is None:
            t = x.new_zeros(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype)
        return super().forward(x, t, y, s=s, mask=mask)


def pixeldit_b(num_classes=1000, **kwargs):
    return PixDiT(
        in_channels=3,
        num_groups=12,
        hidden_size=768,
        pixel_hidden_size=16,
        patch_depth=12,
        pixel_depth=4,
        patch_size=16,
        num_classes=num_classes,
        **kwargs,
    )


def pixeldit_xl(num_classes=1000, **kwargs):
    return PixDiT(
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=16,
        patch_depth=26,
        pixel_depth=4,
        patch_size=16,
        num_classes=num_classes,
        **kwargs,
    )
