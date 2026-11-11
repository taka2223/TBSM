"""Read the existing, unscaled SD-VAE posterior mean/std NPY cache."""
import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class ImageNetMomentsDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        identity = manifest["identity"]
        if not manifest["complete"] or identity["layout"] != "posterior_mean_then_std_unscaled":
            raise ValueError("Require a complete unscaled posterior mean/std cache")
        if identity["shape"] != [8, 32, 32] or identity["classes"] != 1000:
            raise ValueError("Expected ImageNet-256 SD-VAE moments")
        self.shards = manifest["shards"]
        self.starts = [s["start"] for s in self.shards]
        self.count = manifest["encoded"]
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r")
        self.open = OrderedDict()
        cursor = 0
        for shard in self.shards:
            if shard["start"] != cursor or not (self.root / shard["file"]).is_file():
                raise ValueError("Missing or noncontiguous cache shard")
            cursor += shard["count"]
        if cursor != self.count or self.count != identity["count"] or len(self.labels) != self.count:
            raise ValueError("Cache count mismatch")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        shard_id = bisect.bisect_right(self.starts, index) - 1
        if shard_id not in self.open:
            self.open[shard_id] = np.load(self.root / self.shards[shard_id]["file"], mmap_mode="r")
        self.open.move_to_end(shard_id)
        while len(self.open) > 8:
            self.open.popitem(last=False)
        value = np.array(self.open[shard_id][index - self.starts[shard_id]], copy=True)
        return torch.from_numpy(value), int(self.labels[index])

    def __getstate__(self):
        return {**self.__dict__, "open": OrderedDict()}


class ResumableBatches:
    """Deterministic shuffled epochs, exactly replayable across process resumes."""
    def __init__(self, count, batch_size, seed, start_batch=0):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.start_batch = start_batch

    def __iter__(self):
        per_epoch = self.count // self.batch_size
        epoch, offset = divmod(self.start_batch, per_epoch)
        while True:
            rng = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.count, generator=rng)
            for batch in range(offset, per_epoch):
                yield order[batch*self.batch_size:(batch+1)*self.batch_size].tolist()
            epoch, offset = epoch + 1, 0
