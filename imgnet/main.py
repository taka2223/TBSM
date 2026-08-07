import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from builders import build_training_model, module_state_dict
from builders.common import input_shape_from_config, load_config
from builders.datasets import build_dataset


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process(rank):
    return rank == 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_deterministic_backend():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def visual_grid_spec(resolution, num_classes):
    """Return fixed class ids and samples per class for training demos."""
    if resolution >= 512:
        preferred_classes = [29, 291, 628, 985]
        samples_per_class = 4
    elif resolution >= 256:
        preferred_classes = [42, 281, 517, 867]
        samples_per_class = 4
    else:
        class_count = min(num_classes, 10)
        preferred_classes = torch.linspace(0, num_classes - 1, class_count).long().tolist()
        samples_per_class = 10

    if any(class_id >= num_classes for class_id in preferred_classes):
        class_count = min(len(preferred_classes), num_classes)
        preferred_classes = torch.linspace(0, num_classes - 1, class_count).long().tolist()

    return preferred_classes, samples_per_class


def save_sec_per_step(save_dir, step, steps_done, step_sec_sum):
    """Write/overwrite the global average sec/step under save_dir.

    Recomputes avg = step_sec_sum / steps_done so it reflects the current
    global progress, refreshed each time a visual demo grid is saved.
    """
    avg_sec_per_step = step_sec_sum / steps_done if steps_done > 0 else 0.0
    speed_path = os.path.join(save_dir, "sec_per_step.json")
    with open(speed_path, "w") as f:
        json.dump(
            {
                "step": step,
                "steps_done": steps_done,
                "total_sec": step_sec_sum,
                "avg_sec_per_step": avg_sec_per_step,
            },
            f,
            indent=2,
        )
    return speed_path, avg_sec_per_step


def main():
    parser = argparse.ArgumentParser(description="Config-driven training script")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config under configs/")
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    method_cfg = cfg["method"]
    network_cfg = cfg["network"]
    train_cfg = cfg["training"]
    runtime_cfg = cfg["runtime"]

    set_seed(runtime_cfg["seed"])
    if runtime_cfg["deterministic"]:
        configure_deterministic_backend()
    rank, world_size, local_rank = setup_distributed()

    os.makedirs(exp_cfg["save_dir"], exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    runtime_cfg.update({
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    })
    use_bf16 = train_cfg["bf16"] and device.type == "cuda"

    dataset, num_classes = build_dataset(cfg)
    data_cfg["num_classes"] = num_classes
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=runtime_cfg["seed"],
        )
    else:
        sampler = None
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(runtime_cfg["seed"] + rank)

    dataloader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=(sampler is None),
        num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True, sampler=sampler,
        generator=dataloader_generator,
    )

    method_model, start_step = build_training_model(cfg)
    if is_main_process(rank):
        model_total, model_trainable = count_parameters(method_model)
        backbone_total, backbone_trainable = count_parameters(method_model.generator.backbone)
        print(
            "Run config:\n"
            f"  device: {device}\n"
            f"  world_size: {world_size}\n"
            f"  method: {method_cfg['name']}\n"
            f"  network: {network_cfg['name']}\n"
            f"  method_model: {model_total/1e6:.1f}M params ({model_trainable/1e6:.1f}M trainable)\n"
            f"  generator.backbone: {backbone_total/1e6:.1f}M params ({backbone_trainable/1e6:.1f}M trainable)\n"
            f"  dataset: {data_cfg['name']} @ {data_cfg['resolution']}\n"
            f"  bf16: {use_bf16}\n"
            f"  seed: {runtime_cfg['seed']}\n"
            f"  deterministic: {runtime_cfg['deterministic']}\n"
            f"  save_dir: {exp_cfg['save_dir']}"
        )

    if is_main_process(rank):
        vis_classes, vis_cols = visual_grid_spec(data_cfg["resolution"], num_classes)
        vis_batch = len(vis_classes) * vis_cols
        fixed_noise = torch.randn(vis_batch, *input_shape_from_config(cfg), device=device)
        vis_classes = torch.tensor(vis_classes, device=device, dtype=torch.long)
        fixed_y = torch.repeat_interleave(vis_classes, repeats=vis_cols)

    if is_main_process(rank):
        print(f"Beginning {method_cfg['name']} training ({train_cfg['total_steps']} steps from step {start_step}, "
              f"batch_size_per_gpu={train_cfg['batch_size']}, effective_batch={train_cfg['batch_size'] * world_size})...")

    data_iter = iter(dataloader)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    epoch = 0

    # Global accumulators for sec/step over the whole run so far.
    step_sec_sum = 0.0
    step_sec_count = 0

    if train_cfg["vis_every"] > 0 and is_main_process(rank):
        with autocast_ctx:
            generated = method_model.sample(y=fixed_y, noise=fixed_noise)
        path = os.path.join(exp_cfg["save_dir"], f"visual_step_{start_step}.png")
        save_image(generated, path, nrow=vis_cols)
        print(f"  >> Visual grid saved to {path}")

    for step in range(start_step, train_cfg["total_steps"]):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        try:
            images, labels = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        x = images.to(device)
        y = labels.to(device)

        with autocast_ctx:
            loss_dict = method_model.step(x, y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_sec = time.perf_counter() - step_start

        # Accumulate global step timing (every rank tracks locally; rank 0 writes).
        step_sec_sum += step_sec
        step_sec_count += 1

        if is_main_process(rank) and step % train_cfg["log_every"] == 0:
            parts = [f"step_sec={step_sec:.3f}"]
            for k, v in loss_dict.items():
                v = v.item() if hasattr(v, "item") else v
                parts.append(f"{k}={v:.4f}")
            print(f"[Step {step:5d}] " + "  ".join(parts))

        completed_step = step + 1
        if train_cfg["vis_every"] > 0 and completed_step % train_cfg["vis_every"] == 0 and is_main_process(rank):
            with autocast_ctx:
                generated = method_model.sample(y=fixed_y, noise=fixed_noise)
            path = os.path.join(exp_cfg["save_dir"], f"visual_step_{completed_step}.png")
            save_image(generated, path, nrow=vis_cols)
            print(f"  >> Visual grid saved to {path}")

            speed_path, avg_sec_per_step = save_sec_per_step(
                exp_cfg["save_dir"], completed_step, step_sec_count, step_sec_sum
            )
            print(f"  >> Global avg sec/step updated to {avg_sec_per_step:.4f} -> {speed_path}")

        if train_cfg["save_every"] > 0 and (step + 1) % train_cfg["save_every"] == 0 and is_main_process(rank):
            backbone_sd = module_state_dict(method_model.generator.backbone)
            ckpt_path = os.path.join(exp_cfg["save_dir"], f"{method_cfg['name']}_step_{step+1}.pt")
            torch.save({"backbone": backbone_sd, "model": method_model.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  >> Checkpoint saved to {ckpt_path}")

    if is_main_process(rank):
        print("Training finished.")
        # Final global average sec/step over the whole run.
        speed_path, avg_sec_per_step = save_sec_per_step(
            exp_cfg["save_dir"], train_cfg["total_steps"], step_sec_count, step_sec_sum
        )
        print(f"  >> Final global avg sec/step: {avg_sec_per_step:.4f} saved to {speed_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
