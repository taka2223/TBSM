import copy
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from networks import (
    UniversalAdapter,
    jit_b,
    jit_h,
    jit_l,
    pixeldit_b,
    pixeldit_xl,
    dit_xl_2,
    dit_xl_4,
)


def build_network(cfg):
    network_cfg = cfg["network"]
    params = copy.deepcopy(network_cfg["params"])
    adapter_cfg = copy.deepcopy(network_cfg.get("adapter"))
    if adapter_cfg is None:
        adapter_cfg = params.pop("adapter", None)
    network_name = network_cfg["name"]
    num_classes = cfg["data"]["num_classes"]

    if network_name == "pixeldit_b":
        backbone = pixeldit_b(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "pixeldit_xl":
        backbone = pixeldit_xl(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "jit_b":
        backbone = jit_b(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "jit_l":
        backbone = jit_l(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "jit_h":
        backbone = jit_h(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "dit_xl_2":
        backbone = dit_xl_2(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone
    if network_name == "dit_xl_4":
        backbone = dit_xl_4(num_classes=num_classes, **params)
        return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone

    raise ValueError(f"Unknown network: {network_name}")


def input_shape_from_config(cfg):
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        return (
            int(latent_cfg.get("channels", 4)),
            int(cfg["data"]["resolution"]) // int(latent_cfg.get("downsample_ratio", 8)),
            int(cfg["data"]["resolution"]) // int(latent_cfg.get("downsample_ratio", 8)),
        )
    resolution = cfg["data"]["resolution"]
    return (3, resolution, resolution)


def device_from_config(cfg):
    return cfg["runtime"]["device"]


def wrap_ddp_if_needed(method_model, cfg):
    runtime_cfg = cfg["runtime"]
    if runtime_cfg["world_size"] <= 1:
        return method_model

    local_rank = runtime_cfg["local_rank"]
    method_model.generator.backbone = DDP(
        method_model.generator.backbone,
        device_ids=[local_rank],
    )
    for matcher in method_model.matchers:
        if matcher.tracker is not None:
            matcher.tracker = DDP(matcher.tracker, device_ids=[local_rank])
    return method_model


def raw_module(module):
    return module.module if isinstance(module, DDP) else module


def module_state_dict(module):
    return raw_module(module).state_dict()


def extract_backbone_state(state):
    prefixes = [
        "generator.backbone.",
        "_ema_generator.backbone.",
        "_eval_generator.backbone.",
        "backbone.",
    ]
    extracted = {}
    for key, value in state.items():
        name = key
        for prefix in prefixes:
            if key.startswith(prefix):
                name = key[len(prefix):]
                break
        extracted[name] = value
    return extracted


def load_backbone_state(backbone, state):
    raw_backbone = raw_module(backbone)
    try:
        result = raw_backbone.load_state_dict(state, strict=False)
        if len(result.unexpected_keys) == 0:
            return result
    except RuntimeError:
        pass

    prefixed = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in state.items()
    }
    result = raw_backbone.load_state_dict(prefixed, strict=False)
    if len(result.unexpected_keys) == 0:
        return result

    unprefixed = {
        key[len("model."):] if key.startswith("model.") else key: value
        for key, value in state.items()
    }
    return raw_backbone.load_state_dict(unprefixed, strict=False)


def select_init_state(ckpt):
    if "backbone" in ckpt:
        return ckpt["backbone"]
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_checkpoint(method_model, cfg):
    exp_cfg = cfg["experiment"]
    device = device_from_config(cfg)

    if exp_cfg["resume"] and os.path.isfile(exp_cfg["resume"]):
        ckpt = torch.load(exp_cfg["resume"], map_location=device, weights_only=True)
        method_model.load_state_dict(ckpt["model"], strict=False)
        method_model.sync_ema_shadow()
        return ckpt.get("step", "unknown")

    if exp_cfg["init_weights"] and os.path.isfile(exp_cfg["init_weights"]):
        ckpt = torch.load(exp_cfg["init_weights"], map_location=device, weights_only=True)
        state = select_init_state(ckpt)
        backbone_state = extract_backbone_state(state)
        result = load_backbone_state(method_model.generator.backbone, backbone_state)
        if result.missing_keys or result.unexpected_keys:
            print(
                "Loaded init_weights with "
                f"{len(result.missing_keys)} missing keys and "
                f"{len(result.unexpected_keys)} unexpected keys"
            )
        method_model.reset_ema()

    return 0


def finish_build(method_model, cfg):
    start_step = load_checkpoint(method_model, cfg)
    method_model = wrap_ddp_if_needed(method_model, cfg)
    return method_model, start_step
