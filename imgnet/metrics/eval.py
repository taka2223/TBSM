#!/usr/bin/env python3
"""Evaluate TBSM checkpoints with FD, FDr, IS, P&R, and CMMD metrics.

The evaluator builds a one-step generator from a TBSM YAML configuration and
uses distributed feature extraction for ImageNet evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
import numpy as np

from metrics import eval_all_fds
from metrics.utils.distributed_util import get_global_rank, get_world_size

V2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_MODELS = [
    "inception",
    "vit_large_patch16_224.mae",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch14_clip_224.openai",
    "vit_so400m_patch16_siglip_256.v2_webli",
    "convnext",
]


def _abs_path(path: str | None, base: Path) -> str | None:
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _clone_yaml_obj(obj):
    return yaml.safe_load(yaml.safe_dump(obj))


def _import_v2_builders():
    if str(V2_ROOT) not in sys.path:
        sys.path.insert(0, str(V2_ROOT))
    from builders import build_training_model

    return build_training_model


class NoopEMA:
    labels = ["online"]

    @contextlib.contextmanager
    def swap(self, model, label="online"):
        yield


class V2TBSMGenerator(torch.nn.Module):
    def __init__(self, cfg: dict, checkpoint: str | None, weights: str, eval_args):
        super().__init__()
        build_training_model = _import_v2_builders()

        cfg = _clone_yaml_obj(cfg)
        # Evaluation only needs the generator and EMA weights. Training matchers
        # can pull external repr checkpoints, so keep them out of eval builds.
        cfg.setdefault("method", {}).setdefault("params", {})["matchers"] = []
        runtime = cfg.setdefault("runtime", {})
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        runtime["device"] = device
        runtime["rank"] = int(os.environ.get("RANK", 0))
        runtime["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
        runtime["local_rank"] = local_rank

        exp_cfg = cfg.setdefault("experiment", {})
        if checkpoint:
            if weights == "online":
                exp_cfg["init_weights"] = checkpoint
                exp_cfg["resume"] = ""
            elif weights == "ema":
                exp_cfg["resume"] = checkpoint
                exp_cfg["init_weights"] = ""
            else:
                raise ValueError(
                    f"evaluation.weights must be 'online' or 'ema', got {weights!r}"
                )

        self.image_size = int(cfg["data"]["resolution"])
        latent_cfg = cfg.get("latent")
        if latent_cfg is None:
            self.noise_shape = (3, self.image_size, self.image_size)
        else:
            ratio = int(latent_cfg.get("downsample_ratio", 8))
            self.noise_shape = (
                int(latent_cfg.get("channels", 4)),
                self.image_size // ratio,
                self.image_size // ratio,
            )
        self.method_model, self.current_step = build_training_model(cfg)
        self.method_model.eval()

    @torch.inference_mode()
    def generate(self, n_samples, cfg, labels, args, verbose=False, z_t=None):
        del cfg, args, verbose
        if z_t is None:
            z_t = torch.randn(n_samples, *self.noise_shape, device=labels.device, dtype=torch.float32)
        # eval_all_fds expects raw images in [-1, 1], whereas TBSM.sample()
        # returns decoded images in [0, 1]. Convert back to its input contract.
        samples = self.method_model.sample(labels.to(labels.device), z_t.to(labels.device))
        return samples * 2.0 - 1.0


def _latest_checkpoint(save_dir: str, method_name: str) -> str:
    pattern = f"{method_name}_step_*.pt"
    candidates = list(Path(save_dir).glob(pattern))
    if not candidates:
        return ""

    def step(path: Path):
        stem = path.stem
        try:
            return int(stem.rsplit("_step_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return str(max(candidates, key=step))


def _build_eval_args(config_path: str, cfg: dict) -> argparse.Namespace:
    eval_cfg = cfg.get("evaluation", {}) or {}
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    save_dir = _abs_path(exp_cfg["save_dir"], V2_ROOT)
    checkpoint = eval_cfg.get("checkpoint") or exp_cfg.get("resume")
    if not checkpoint:
        checkpoint = _latest_checkpoint(save_dir, cfg["method"]["name"])
    checkpoint_abs = _abs_path(checkpoint, V2_ROOT) if checkpoint else ""
    if not checkpoint_abs:
        raise FileNotFoundError(
            f"No checkpoint configured or found for eval config: {config_path}"
        )
    if not os.path.isfile(checkpoint_abs):
        raise FileNotFoundError(
            f"Eval checkpoint does not exist: {checkpoint_abs}"
        )
    eval_name = eval_cfg.get("name") or (Path(checkpoint_abs).stem if checkpoint_abs else "latest")
    eval_dir = _abs_path(
        eval_cfg.get("save_dir", os.path.join(save_dir, "eval", eval_name)),
        V2_ROOT,
    )

    eval_bsz = int(eval_cfg.get("batch_size", eval_cfg.get("eval_bsz", 256)))
    return argparse.Namespace(
        config=config_path,
        load_from=checkpoint_abs,
        weights=str(eval_cfg["weights"]),
        output_dir=eval_dir,
        overwrite_output_dir=bool(eval_cfg.get("overwrite", True)),
        log_dir=eval_dir,

        input_size=int(eval_cfg.get("input_size", data_cfg.get("resolution", 256))),
        num_classes=int(eval_cfg.get("num_classes", data_cfg.get("num_classes", 1000))),
        eval_bsz=eval_bsz,
        num_images=int(eval_cfg.get("num_images", 50000)),
        cfg_list=[1.0],
        models=[str(x) for x in eval_cfg.get("models", DEFAULT_EVAL_MODELS)],
        eval_ema_labels=None,
        current_step=0,

        num_sampling_steps=1,
        sampling_method="euler",
        interval_min=0.0,
        interval_max=1.0,
        same_noise=False,
        seed=int(eval_cfg.get("seed", cfg.get("runtime", {}).get("seed", 1))),
        dtype=str(eval_cfg.get("dtype", "bf16")),

        no_prc=bool(eval_cfg.get("no_prc", False)),
        eval_mmd=bool(eval_cfg.get("eval_mmd", False)),
        prc_models=None if eval_cfg.get("prc_models") is None else [str(x) for x in eval_cfg["prc_models"]],
        prc_ref_dir=str(eval_cfg.get(
            "prc_ref_dir",
            os.environ.get("DM_DATA_ZOO", "/path/to/your/data_zoo") + "/imagenet-val-prc",
        )),
        mmd_ref_dir=str(eval_cfg.get(
            "mmd_ref_dir",
            os.environ.get("DM_DATA_ZOO", "/path/to/your/data_zoo") + "/imagenet/val",
        )),
        prc_k=int(eval_cfg.get("prc_k", 3)),
        prc_batch_size=int(eval_cfg.get("prc_batch_size", 10000)),

        class_of_interest=[207, 360, 387, 974, 88, 979, 417, 279],
        force_class_of_interest=False,
    )


def parse_args():
    launcher = argparse.ArgumentParser(description="Evaluate a TBSM YAML config")
    launcher.add_argument("config", help="Path to a TBSM YAML config")
    launch_args = launcher.parse_args()
    config_path = _abs_path(launch_args.config, Path.cwd())
    cfg = _load_yaml(config_path)
    return _build_eval_args(config_path, cfg), cfg


def setup_eval(args):
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    rank, world_size = get_global_rank(), get_world_size()
    args.rank = rank
    args.world_size = world_size
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if rank == 0 and args.overwrite_output_dir and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()


def main():
    args, cfg = parse_args()
    setup_eval(args)
    model = V2TBSMGenerator(cfg, args.load_from, args.weights, args).cuda().eval()
    args.current_step = model.current_step
    eval_all_fds.main_generate(args, model, NoopEMA())


if __name__ == "__main__":
    main()
