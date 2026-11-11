import copy
import os

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from networks import UniversalAdapter, create_backbone


def _expand_config_values(value):
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping, got {type(config).__name__}")
    return _expand_config_values(config)


def require_existing_path(path, config_key, *, path_type=None):
    if not path:
        raise ValueError(f"{config_key} must not be empty")
    if path_type == "file":
        exists = os.path.isfile(path)
        expected = "file"
    elif path_type == "directory":
        exists = os.path.isdir(path)
        expected = "directory"
    else:
        exists = os.path.exists(path)
        expected = "path"
    if not exists:
        raise FileNotFoundError(
            f"Configured {config_key} {expected} does not exist: {path}"
        )
    return path


def validate_checkpoint_paths(cfg):
    exp_cfg = cfg["experiment"]
    resume_path = exp_cfg.get("resume")
    init_weights = exp_cfg.get("init_weights")
    if resume_path:
        require_existing_path(resume_path, "experiment.resume", path_type="file")
    elif init_weights:
        require_existing_path(init_weights, "experiment.init_weights", path_type="file")
    return resume_path, init_weights


def build_network(cfg):
    network_cfg = cfg["network"]
    params = copy.deepcopy(network_cfg["params"])
    adapter_cfg = copy.deepcopy(network_cfg.get("adapter"))
    if adapter_cfg is None:
        adapter_cfg = params.pop("adapter", None)
    network_name = network_cfg["name"]
    num_classes = cfg["data"]["num_classes"]

    backbone = create_backbone(network_name, num_classes=num_classes, **params)
    return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone


def input_shape_from_config(cfg):
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        return (
            int(latent_cfg.get("channels", 4)),
            int(cfg["data"]["resolution"])
            // int(latent_cfg.get("downsample_ratio", 8)),
            int(cfg["data"]["resolution"])
            // int(latent_cfg.get("downsample_ratio", 8)),
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
    for field in method_model.representation_fields:
        if field.tracker is not None:
            field.tracker = DDP(field.tracker, device_ids=[local_rank])
    return method_model


def raw_module(module):
    return module.module if isinstance(module, DDP) else module


def module_state_dict(module):
    return raw_module(module).state_dict()


def extract_backbone_state(state):
    prefixes = [
        "generator.backbone.",
        "_eval_generator.backbone.",
        "backbone.",
    ]
    extracted = {}
    for key, value in state.items():
        name = key
        for prefix in prefixes:
            if key.startswith(prefix):
                name = key[len(prefix) :]
                break
        extracted[name] = value
    return extracted


def load_backbone_state(backbone, state):
    raw_backbone = raw_module(backbone)
    candidates = [state]
    for prefix in ("model.", "backbone.", "backbone.model."):
        candidates.append(
            {
                key if key.startswith(prefix) else f"{prefix}{key}": value
                for key, value in state.items()
            }
        )
    candidates.append(
        {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state.items()
        }
    )

    best_result = None
    best_score = None
    for candidate in candidates:
        try:
            result = raw_backbone.load_state_dict(candidate, strict=False)
        except RuntimeError:
            continue
        score = len(result.missing_keys) + len(result.unexpected_keys)
        if best_score is None or score < best_score:
            best_result = result
            best_score = score
        if score == 0:
            break
    if best_result is None:
        raise RuntimeError("Unable to load checkpoint into the configured backbone")
    return best_result


def select_init_state(ckpt):
    if "backbone" in ckpt:
        return ckpt["backbone"]
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_checkpoint(method_model, cfg):
    device = device_from_config(cfg)
    resume_path, init_weights = validate_checkpoint_paths(cfg)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        result = method_model.load_state_dict(ckpt["model"], strict=False)
        unexpected = list(result.unexpected_keys)
        if not method_model.representation_fields:
            unexpected = [
                key
                for key in unexpected
                if not key.startswith("representation_fields.")
            ]
        if result.missing_keys or unexpected:
            raise RuntimeError(
                "Resume checkpoint does not match the current TBSM model: "
                f"missing={result.missing_keys}, unexpected={unexpected}"
            )
        return ckpt.get("step", "unknown")

    if init_weights:
        ckpt = torch.load(init_weights, map_location="cpu", weights_only=True)
        state = select_init_state(ckpt)
        backbone_state = extract_backbone_state(state)
        result = load_backbone_state(method_model.generator.backbone, backbone_state)
        if result.missing_keys or result.unexpected_keys:
            if cfg["experiment"].get("strict_init", False):
                raise RuntimeError(
                    f"Incomplete initialization: missing={result.missing_keys}, "
                    f"unexpected={result.unexpected_keys}"
                )
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
