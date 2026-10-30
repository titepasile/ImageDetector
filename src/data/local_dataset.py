"""Read the converter's tar shards as a PyTorch dataset.

Each sample is one 128x128 crop. It is returned as
    (image, label, weight, image_id)
  image     float tensor 3x128x128, normalised to roughly [-1, 1]
  label     0 = real (nature), 1 = AI-generated
  weight    1 / number of crops of that image, so every IMAGE counts the same
            in the loss (BigGAN images are 128x128 and give only 1 crop,
            larger images give 4)
  image_id  string, used to average the crops of one image at test time

Why shuffling happens in three places:
  The converter walks the GenImage folders in order, so long stretches of
  shards hold only one generator and one class. To still get mixed batches:
    1. the shard order is shuffled every epoch,
    2. every worker reads from several shards at once (`interleave`) and
       picks the next sample from a random one of them,
    3. samples pass through a shuffle buffer (`buffer_size`).
  For evaluation all three are switched off and the order is deterministic.

Nothing here resizes an image. Augmentations are only flips (and optional
JPEG re-compression), because anything that interpolates pixels would
destroy the fine traces the local encoder is supposed to learn.
"""

import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def list_shards(root, split):
    """Return (shard paths, number of crops) for one split ('train'/'val'/'test')."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    paths = sorted((root / split).glob("*.tar"))
    if not paths:
        raise FileNotFoundError(f"No shards in {root / split}")
    total = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        counts = {shard["name"]: shard["samples"] for shard in manifest["shards"]}
        total = sum(counts.get(f"{split}/{p.name}", 0) for p in paths)
    return [str(p) for p in paths], total


def iter_shard(path):
    """Yield (png_bytes, metadata) pairs from one shard, in file order."""
    pending = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, extension = member.name.rsplit(".", 1)
            data = archive.extractfile(member).read()
            entry = pending.setdefault(key, {})
            entry[extension] = data
            if "png" in entry and "json" in entry:
                del pending[key]
                yield entry["png"], json.loads(entry["json"])


class Augment:
    """Train-time augmentation. No interpolation, no resizing."""

    def __init__(self, hflip=True, jpeg_prob=0.0, jpeg_quality=(70, 100)):
        self.hflip = hflip
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality = jpeg_quality

    def __call__(self, image, rng):
        if self.jpeg_prob and rng.random() < self.jpeg_prob:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(*self.jpeg_quality))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        if self.hflip and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


def to_tensor(image):
    """uint8 HxWx3 -> float 3xHxW in about [-1, 1]."""
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).sub_(127.5).div_(127.5)


class CropShards(IterableDataset):
    def __init__(self, shards, *, train, augment=None, interleave=8,
                 buffer_size=2000, seed=0, weight_by_image=True):
        super().__init__()
        self.shards = list(shards)
        self.train = train
        self.augment = augment if train else None
        self.interleave = interleave if train else 1
        self.buffer_size = buffer_size if train else 0
        self.seed = seed
        self.weight_by_image = weight_by_image
        self.epoch = 0

    def set_epoch(self, epoch):
        """Call before every epoch so the shuffling differs between epochs."""
        self.epoch = epoch

    def _my_shards(self, rng):
        shards = list(self.shards)
        if self.train:
            rng.shuffle(shards)
        info = get_worker_info()
        if info is None:
            return shards
        return shards[info.id::info.num_workers]

    def _raw_samples(self, shards, rng):
        if self.interleave <= 1:
            for path in shards:
                yield from iter_shard(path)
            return
        queue = list(shards)
        active = []
        while queue or active:
            while queue and len(active) < self.interleave:
                active.append(iter_shard(queue.pop()))
            index = rng.randrange(len(active))
            try:
                yield next(active[index])
            except StopIteration:
                active.pop(index)

    def _shuffled(self, samples, rng):
        if self.buffer_size <= 1:
            yield from samples
            return
        buffer = []
        for sample in samples:
            if len(buffer) < self.buffer_size:
                buffer.append(sample)
                continue
            index = rng.randrange(self.buffer_size)
            yield buffer[index]
            buffer[index] = sample
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        shard_rng = random.Random(f"{self.seed}:{self.epoch}")       # same in all workers
        rng = random.Random(f"{self.seed}:{self.epoch}:{worker}")     # differs per worker
        shards = self._my_shards(shard_rng)
        for png, meta in self._shuffled(self._raw_samples(shards, rng), rng):
            image = Image.open(io.BytesIO(png)).convert("RGB")
            if self.augment is not None:
                image = self.augment(image, rng)
            weight = 1.0 / meta.get("num_crops_of_image", 1) if self.weight_by_image else 1.0
            yield (to_tensor(image), torch.tensor(meta["label"], dtype=torch.long),
                   torch.tensor(weight, dtype=torch.float32), meta["image_id"])


def make_loader(root, split, *, batch_size=128, num_workers=4, train=None,
                jpeg_aug_prob=0.0, seed=0, **dataset_kwargs):
    """DataLoader for one split. train defaults to (split == 'train')."""
    train = split == "train" if train is None else train
    shards, total = list_shards(root, split)
    dataset = CropShards(shards, train=train, seed=seed,
                         augment=Augment(jpeg_prob=jpeg_aug_prob) if train else None,
                         **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=min(num_workers, len(shards)),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=False, drop_last=train)
    return loader, dataset, total
