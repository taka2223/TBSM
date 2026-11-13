import numpy as np
import torch

from methods.common import Generator
from methods.fd_loss import FDLoss
from networks.vaes import SDVAE

from .common import build_network, device_from_config, finish_build, require_existing_path, validate_checkpoint_paths
from .tbsm import build_representor


def build(cfg):
    validate_checkpoint_paths(cfg)
    params = cfg["method"]["params"]
    device = device_from_config(cfg)
    latent_cfg = cfg["latent"]
    require_existing_path(latent_cfg["decoder_path"], "latent.decoder_path", path_type="directory")
    stats_path = require_existing_path(params["reference_stats_path"], "method.params.reference_stats_path", path_type="file")
    representor_cfg = dict(params["representation"])
    ckpt_path = representor_cfg.get("ckpt_path")
    if ckpt_path:
        require_existing_path(ckpt_path, "method.params.representation.ckpt_path")
    representor = build_representor(representor_cfg).to(device)
    with np.load(stats_path) as stats:
        mean = torch.from_numpy(stats["mu"]).to(device=device, dtype=torch.float64)
        covariance = torch.from_numpy(stats["sigma"]).to(device=device, dtype=torch.float64)
    expected = representor.feat_dim
    if mean.shape != (expected,) or covariance.shape != (expected, expected):
        raise ValueError(f"FD reference shape mismatch: mean={tuple(mean.shape)}, covariance={tuple(covariance.shape)}, representor_dim={expected}")
    generator = Generator(
        build_network(cfg).to(device),
        decoder=SDVAE(model_path=latent_cfg["decoder_path"], multiplier=latent_cfg.get("multiplier", 0.18215)).to(device),
    ).to(device)
    trainer = params["trainer"]
    population = params["population"]
    model = FDLoss(
        generator=generator, representor=representor,
        reference_mean=mean, reference_covariance=covariance,
        gen_lr=trainer["gen_lr"], ema_decay=trainer["ema_decay"],
        population_beta=population.get("ema_beta", 0.999),
        fid_norm_eps=params.get("fid_norm_eps", 0.01),
        noise_scale=trainer.get("noise_scale", 1.0),
        gen_betas=trainer.get("gen_betas", (0.9, 0.999)),
        grad_accum_steps=trainer.get("grad_accum_steps", 1),
    ).to(device)
    return finish_build(model, cfg)
