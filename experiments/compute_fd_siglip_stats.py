#!/usr/bin/env python3
"""Compute ImageNet-train SigLIP2 mean/covariance with Dhariwal 256 crop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imgnet"))
from builders.tbsm import build_representor


def dhariwal_crop(image, size=256):
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(value // 2 for value in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(value * scale) for value in image.size), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"))
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    return array[top:top + size, left:left + size]


class ImageNetTrain(Dataset):
    def __init__(self, root, index_path=None):
        self.root = Path(root)
        if index_path:
            self.entries = [item[0] for item in json.loads(Path(index_path).read_text())]
        else:
            classes = sorted(path for path in self.root.iterdir() if path.is_dir())
            if len(classes) != 1000:
                raise ValueError(f"Expected 1000 ImageNet classes, found {len(classes)}")
            Image.init()
            self.entries = [
                str(path.relative_to(self.root))
                for class_dir in classes
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in Image.EXTENSION
            ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        with Image.open(self.root / self.entries[index]) as image:
            array = dhariwal_crop(image)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    dataset = ImageNetTrain(args.root, args.index)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if count < 2:
        parser.error("Need at least two images")
    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(count)), batch_size=args.batch,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    representor = build_representor(dict(
        arch="siglip2_b", input_size=256, ckpt_path=args.model,
        use_post_layernorm=True,
    )).cuda().eval()
    dimension = representor.feat_dim
    total = torch.zeros(dimension, dtype=torch.float64, device="cuda")
    outer = torch.zeros(dimension, dimension, dtype=torch.float64, device="cuda")
    seen = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images in loader:
            features = representor.extract(images.cuda(non_blocking=True)).double()
            total.add_(features.sum(0)); outer.add_(features.T @ features)
            seen += features.shape[0]
            if seen == count or seen % (args.batch * 50) == 0:
                print(json.dumps({"event": "stats_progress", "images": seen, "total": count}), flush=True)
    mean = total / seen
    covariance = outer / seen - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    metadata = dict(
        format="tbsm_fd_siglip2_reference_v1", count=seen, classes=1000,
        crop="center-crop-dhariwal_256", pixel_range="0_1",
        representation="siglip2-base-patch16-256", pooling="mean_patch_tokens",
        post_layernorm=True, covariance="population", source_root=str(Path(args.root).resolve()),
        index_sha256=file_sha256(args.index) if args.index else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, mu=mean.cpu().numpy(), sigma=covariance.cpu().numpy(),
                 metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(output)
    print(json.dumps({"event": "stats_complete", "path": str(output), **metadata}), flush=True)


if __name__ == "__main__":
    main()
