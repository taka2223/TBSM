from methods.common import Generator
from methods.tbsm import RepresentationScatteringField, TBSM
from networks.representors import create_representor
from networks.vaes import SDVAE

from .common import (
    build_network,
    device_from_config,
    finish_build,
    require_existing_path,
    validate_checkpoint_paths,
)


def build_representor(cfg):
    params = dict(cfg)
    arch = params.pop("arch")
    ckpt_path = params.get("ckpt_path")
    if ckpt_path:
        require_existing_path(ckpt_path, "representor.ckpt_path")
    return create_representor(arch, **params)


def build_representors(params, device):
    representor_specs = []
    for field_config in params["representation_fields"]:
        representor_cfg = dict(field_config["representor"])
        loss_weight = representor_cfg.pop("loss_weight")
        representor_specs.append(
            (build_representor(representor_cfg).to(device), loss_weight)
        )
    return representor_specs


def build_representation_fields(params, representation_specs, num_classes, device):
    field_params = {
        "lambda_weight": params["lambda"],
        "rho": params["rho"],
        "num_classes": num_classes,
        "feature_norm": params.get("feature_norm", ["mu", "std", "rms"]),
        "field_config": params.get("field", {"type": "energy"}),
    }
    return [
        RepresentationScatteringField(
            representor=representor,
            **field_params,
            loss_weight=loss_weight,
        ).to(device)
        for representor, loss_weight in representation_specs
    ]


def _validate_asset_paths(cfg):
    validate_checkpoint_paths(cfg)
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        require_existing_path(
            latent_cfg["decoder_path"],
            "latent.decoder_path",
            path_type="directory",
        )
    for field_config in cfg["method"]["params"]["representation_fields"]:
        ckpt_path = field_config["representor"].get("ckpt_path")
        if ckpt_path:
            require_existing_path(ckpt_path, "representor.ckpt_path")


def build(cfg):
    _validate_asset_paths(cfg)
    params = cfg["method"]["params"]
    device = device_from_config(cfg)
    num_classes = cfg["data"]["num_classes"]
    backbone = build_network(cfg).to(device)
    representation_specs = build_representors(params, device)

    decoder = None
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        decoder = SDVAE(
            model_path=latent_cfg["decoder_path"],
            multiplier=latent_cfg.get("multiplier", 0.18215),
        ).to(device)

    generator = Generator(backbone, decoder=decoder).to(device)
    representation_fields = build_representation_fields(
        params, representation_specs, num_classes, device
    )
    model = TBSM(
        generator=generator,
        representation_fields=representation_fields,
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
