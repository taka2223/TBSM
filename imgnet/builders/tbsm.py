from methods.common import Generator
from methods.tbsm import RepresentationScattering, TBSM
from networks.representors import (
    MAERepresentor,
    ResNetRepresentor,
    SigLIP2Representor,
)

from .common import build_network, device_from_config, finish_build
from networks.vaes import SDVAE


def build_representation(cfg):
    params = dict(cfg)
    arch = params.pop("arch")
    if arch == "mae_b":
        return MAERepresentor(arch=arch, **params)
    if arch == "siglip2_b":
        return SigLIP2Representor(arch=arch, **params)
    if arch == "resnet18":
        return ResNetRepresentor(arch=arch, **params)
    raise ValueError(f"Unknown TBSM representor arch={arch!r}")


def build_matchers(params, device):
    matchers_cfg = params.get("matchers")
    if matchers_cfg is None:
        raise KeyError("method.params.matchers is required")

    matchers = []
    for matcher_cfg in matchers_cfg:
        representor = build_representation(matcher_cfg["representor"]).to(device)
        matcher = RepresentationScattering(
            representor=representor,
            **matcher_cfg["params"],
        ).to(device)
        matchers.append(matcher)
    return matchers


def build(cfg):
    params = cfg["method"]["params"]
    device = device_from_config(cfg)
    num_classes = cfg["data"]["num_classes"]
    backbone = build_network(cfg).to(device)
    matchers = build_matchers(params, device)

    decoder = None
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        decoder = SDVAE(
            model_path=latent_cfg["decoder_path"],
            multiplier=latent_cfg.get("multiplier", 0.18215),
        ).to(device)

    generator = Generator(backbone, decoder=decoder).to(device)
    model = TBSM(
        generator=generator,
        matchers=matchers,
        num_classes=num_classes,
        gen_lr=params["trainer"]["gen_lr"],
        tracker_lr=params["trainer"]["tracker_lr"],
        ema_decay=params["trainer"]["ema_decay"],
        t_sampling=params["trainer"]["t_sampling"],
        noise_scale=params["trainer"].get("noise_scale", 1.0),
        gen_betas=params["trainer"].get("gen_betas", (0.9, 0.999)),
        tracker_betas=params["trainer"].get("tracker_betas", (0.9, 0.999)),
        grad_accum_steps=params["trainer"].get("grad_accum_steps", 1),
    ).to(device)
    return finish_build(model, cfg)
