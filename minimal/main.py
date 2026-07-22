"""Config-driven minimal MNIST entry point for TBSM."""

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from methods.generator import Generator
from methods.representor import build_representation
from methods.tbsm import RepresentationScattering, TBSM


def build_loader(config):
    data_config = config["data"]
    training_config = config["training"]
    image_transform = transforms.Compose([
        transforms.Resize(
            (data_config["resolution"], data_config["resolution"])
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    training_dataset = datasets.MNIST(
        data_config["data_dir"],
        train=True,
        download=True,
        transform=image_transform,
    )
    return DataLoader(
        training_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def build_model(config, device):
    generator = Generator(
        num_classes=config["data"]["num_classes"],
        **config["network"].get("params", {}),
    ).to(device)
    representation_fields = []
    for field_config in config["method"]["params"]["matchers"]:
        representation_map = build_representation(
            field_config["representor"]
        ).to(device)
        representation_field = RepresentationScattering(
            representor=representation_map,
            num_classes=config["data"]["num_classes"],
            **field_config["params"],
        ).to(device)
        representation_fields.append(representation_field)

    optimization_config = config["method"]["params"]["trainer"]
    return TBSM(
        generator=generator,
        matchers=representation_fields,
        gen_lr=optimization_config["gen_lr"],
        tracker_lr=optimization_config["tracker_lr"],
        t_sampling=optimization_config["t_sampling"],
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    random.seed(config["runtime"]["seed"])
    torch.manual_seed(config["runtime"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["experiment"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tbsm = build_model(config, device)
    training_loader = build_loader(config)
    data_iterator = iter(training_loader)

    resolution = config["data"]["resolution"]
    fixed_labels = torch.arange(10, device=device).repeat_interleave(10)
    fixed_noise = torch.randn(100, 3, resolution, resolution, device=device)
    training_config = config["training"]
    for step in range(training_config["total_steps"]):
        try:
            real_images, class_labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(training_loader)
            real_images, class_labels = next(data_iterator)
        metrics = tbsm.step(real_images.to(device), class_labels.to(device))

        if step % training_config["log_every"] == 0:
            formatted_metrics = " ".join(
                f"{key}={value.item():.4f}" for key, value in metrics.items()
            )
            print(f"step={step:05d} {formatted_metrics}")
        completed_steps = step + 1
        if (
            training_config["vis_every"]
            and completed_steps % training_config["vis_every"] == 0
        ):
            save_image(
                tbsm.sample(fixed_noise, fixed_labels),
                output_dir / f"samples_{completed_steps}.png",
                nrow=10,
            )
        if (
            training_config["save_every"]
            and completed_steps % training_config["save_every"] == 0
        ):
            torch.save(
                tbsm.state_dict(), output_dir / f"tbsm_{completed_steps}.pt"
            )


if __name__ == "__main__":
    main()
