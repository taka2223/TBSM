#!/usr/bin/env python3
"""Single-GPU ImageNet FD-SigLIP training, checkpointing, and evaluation."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imgnet"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import torch
import yaml

import imagenet_tbsm as common
from builders.fd_loss import build as build_fd_loss
from builders.common import load_config


def setup(cfg):
    seed = cfg["runtime"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    if cfg["method"]["name"] != "fd_loss":
        raise ValueError("imagenet_fd_siglip.py requires method.name=fd_loss")
    if cfg["method"]["params"]["trainer"].get("grad_accum_steps", 1) != 1:
        raise ValueError("FD loss requires grad_accum_steps=1")
    build_cfg = copy.deepcopy(cfg)
    build_cfg["runtime"].update(device=torch.device("cuda:0"), world_size=1, local_rank=0)
    build_cfg["experiment"]["resume"] = ""
    model, _ = build_fd_loss(build_cfg)
    common.emit(
        "initialized", parameters=sum(p.numel() for p in model.generator.parameters()),
        strict_init=cfg["experiment"].get("strict_init"), bf16=cfg["training"]["bf16"],
        method="fd_siglip",
    )
    return model


def save_checkpoint(model, cfg, step):
    out = Path(cfg["experiment"]["save_dir"])
    temporary = out / "latest.pt.tmp"
    torch.save(dict(
        model=model.state_dict(),
        generator_optimizer=model.generator_optimizer.state_dict(),
        step=step, accum_count=model._accum_count,
        rng=common.rng_state(), config=cfg,
    ), temporary)
    temporary.replace(out / "latest.pt")
    if step > 0 and step % cfg["evaluation"]["every"] == 0:
        torch.save(
            dict(backbone=model._eval_generator.backbone.state_dict(), step=step),
            out / f"ema_step_{step:07d}.pt",
        )
    common.atomic_json(out / "status.json", dict(step=step, checkpoint="latest.pt"))
    common.emit("checkpoint", step=step, path=str(out / "latest.pt"))


def resume(model, cfg):
    path = Path(cfg["experiment"]["save_dir"]) / "latest.pt"
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    old_cfg, new_cfg = copy.deepcopy(state["config"]), copy.deepcopy(cfg)
    for candidate in (old_cfg, new_cfg):
        candidate["training"].pop("total_steps", None)
    if old_cfg != new_cfg:
        raise ValueError("Refusing FD resume with a changed experiment protocol")
    model.load_state_dict(state["model"], strict=True)
    model.generator_optimizer.load_state_dict(state["generator_optimizer"])
    model._accum_count = state["accum_count"]
    common.restore_rng(state["rng"])
    common.emit("resumed", step=state["step"], population_count=int(model.population.count))
    return int(state["step"])


def initialize_population(model, cfg):
    population = cfg["method"]["params"]["population"]
    common.emit(
        "fd_population_start", count=population["size"],
        batch_size=population["fill_batch_size"], seed=population["seed"],
    )

    def progress(images, total):
        if images == total or images % (population["fill_batch_size"] * 50) == 0:
            common.emit("fd_population_progress", images=images, total=total)

    with common.autocast(cfg):
        model.initialize_population(
            population["size"], population["fill_batch_size"],
            population["seed"], progress=progress,
        )
    common.emit("fd_population_complete", count=int(model.population.count.item()))


def train(cfg, until=None):
    out = Path(cfg["experiment"]["save_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = setup(cfg)
    step = resume(model, cfg)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if not model.population.initialized:
        initialize_population(model, cfg)
        save_checkpoint(model, cfg, 0)
    data = common.loader(cfg, start=step)
    end = min(until or cfg["training"]["total_steps"], cfg["training"]["total_steps"])
    if step == 0:
        common.visualize(model, cfg, 0)
    if step and step % cfg["evaluation"]["every"] == 0:
        metric = out / "metrics" / f"fid{cfg['evaluation']['num_images']}_step_{step:07d}.json"
        if not metric.exists():
            common.fid(model, cfg, step)
    with (out / "history.jsonl").open("a", buffering=1) as history:
        while step < end:
            started = time.perf_counter()
            x, y = common.real_batch(cfg, data)
            with common.autocast(cfg):
                logs = common.finite_logs(model.step(x, y))
            step += 1
            torch.cuda.synchronize()
            record = dict(step=step, seconds=time.perf_counter() - started, **logs)
            if step <= 5 or step % cfg["training"]["log_every"] == 0:
                history.write(json.dumps(record) + "\n")
                common.emit("train", **record)
            if step % cfg["training"]["vis_every"] == 0:
                common.visualize(model, cfg, step)
            if step % cfg["training"]["save_every"] == 0 or step == end:
                save_checkpoint(model, cfg, step)
            if step % cfg["evaluation"]["every"] == 0:
                common.fid(model, cfg, step)


def evaluate(cfg, checkpoint, count):
    model = setup(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model._eval_generator.backbone.load_state_dict(state["backbone"], strict=True)
    model._eval_generator.eval()
    return common.fid(model, cfg, int(state["step"]), count=count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--until", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode == "train":
        train(cfg, args.until)
    else:
        if not args.checkpoint or not args.count:
            parser.error("eval requires --checkpoint and --count")
        evaluate(cfg, args.checkpoint, args.count)


if __name__ == "__main__":
    main()
