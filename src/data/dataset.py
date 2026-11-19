"""Explicit split manifests and deterministic synthetic integration fixtures.

Real manifests are JSONL: path, label (0/1), split, group_id, source_domain,
and generator_family for generated images. Relative paths use the manifest's
directory. No split or label is guessed from a filename.
"""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .views import ImageViews, canonical_image


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}. Supply data.manifest=... "
            "or use data=synthetic for an integration smoke run."
        )
    records = []
    groups, contents, ids = {}, {}, set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"path", "label", "split", "group_id", "source_domain"}
        if not required <= record.keys():
            raise ValueError(f"Manifest line {line_number} needs {sorted(required)}")
        if type(record["label"]) is not int or record["label"] not in (0, 1):
            raise ValueError(f"Invalid label on manifest line {line_number}")
        if any(
            not isinstance(record[key], str) or not record[key]
            for key in ("split", "group_id", "source_domain")
        ):
            raise ValueError(f"Empty split/group/domain on line {line_number}")
        if record["label"] == 1 and not record.get("generator_family"):
            raise ValueError(
                f"Generated sample on line {line_number} needs generator_family"
            )
        image_path = Path(record["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        record["path"] = str(image_path.resolve())
        digest = file_digest(image_path)
        if record.get("sha256", digest) != digest:
            raise ValueError(f"Image checksum mismatch: {image_path}")
        record["sha256"] = digest
        record.setdefault("sample_id", digest)
        if record["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        ids.add(record["sample_id"])
        for seen, key in ((groups, record["group_id"]), (contents, digest)):
            if key in seen and seen[key] != record["split"]:
                raise ValueError(f"Data leakage across splits on line {line_number}")
            seen[key] = record["split"]
        records.append(record)
    if not records:
        raise ValueError("Dataset manifest is empty")
    return records


def group_weights(records):
    """Equal total weight per class, per domain/family, then per image group."""
    strata, groups, counts = {}, {}, Counter()
    for record in records:
        label = record["label"]
        stratum = record["generator_family"] if label else record["source_domain"]
        strata.setdefault(label, set()).add(stratum)
        groups.setdefault((label, stratum), set()).add(record["group_id"])
        counts[label, stratum, record["group_id"]] += 1
    weights = []
    for record in records:
        label = record["label"]
        stratum = record["generator_family"] if label else record["source_domain"]
        denominator = (
            len(strata)
            * len(strata[label])
            * len(groups[label, stratum])
            * counts[label, stratum, record["group_id"]]
        )
        weights.append(len(records) / denominator)
    return weights


def augment_image(image, settings, seed_text):
    """Blur and JPEG re-compression (Wang et al., 2020), each with probability p.

    Deterministic for a given seed text, so worker count does not change results.
    """
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8]))
    if rng.random() < settings.probability:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*settings.blur_sigma)))
    if rng.random() < settings.probability:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=rng.randint(*settings.jpeg_quality))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
    return image


class DetectionDataset(Dataset):
    def __init__(
        self, records, views, synthetic=False, image_size=320, augment=None
    ):
        if not records:
            raise ValueError("The requested dataset split contains no samples")
        self.records = records
        self.views = views
        self.synthetic = synthetic
        self.image_size = image_size
        enabled = augment is not None and augment.probability > 0
        self.augment = augment if enabled and views.training else None
        self.weights = (
            group_weights(records) if views.training else [1.0] * len(records)
        )

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.views.epoch = epoch

    def __getitem__(self, index):
        record = self.records[index]
        if self.synthetic:
            rng = np.random.default_rng(record["image_seed"])
            side = (
                self.image_size
                if record["image_seed"] % 5
                else max(16, self.views.local_size // 2)
            )
            pixels = rng.integers(40, 210, (side, side, 3), dtype=np.uint8)
            if record["label"]:
                pixels[::2, :, 0] = (
                    240  # Deliberate toy signal, not simulated forensic evidence.
                )
            image = Image.fromarray(pixels)
        else:
            with Image.open(record["path"]) as original:
                image = canonical_image(original)
        if self.augment is not None:
            seed_text = f"{self.views.seed}:{self.views.epoch}:{record['sample_id']}:augment"
            image = augment_image(image, self.augment, seed_text)
        batch = self.views(image, record["sample_id"])
        batch.update(
            target=record["label"],
            loss_weight=self.weights[index],
            sample_id=record["sample_id"],
            group_id=record["group_id"],
            source_domain=record["source_domain"],
            generator_family=record.get("generator_family") or "real",
        )
        return batch


def build_datasets(config, real_only=False, evaluation_split=None):
    split_names = (
        config.data.train_split,
        config.data.val_split,
        config.data.test_split,
    )
    if len(set(split_names)) != 3:
        raise ValueError("Train, validation, and test split names must be distinct")
    if config.data.name == "synthetic":
        records = []
        for split_index, (split, count) in enumerate(
            (
                (config.data.train_split, config.data.train_samples),
                (config.data.val_split, config.data.val_samples),
                (config.data.test_split, config.data.test_samples),
            )
        ):
            for index in range(count):
                sample_id = f"synthetic-{split}-{index}"
                records.append(
                    dict(
                        sample_id=sample_id,
                        group_id=sample_id,
                        split=split,
                        label=index % 2,
                        source_domain="synthetic_fixture",
                        generator_family="toy" if index % 2 else None,
                        image_seed=config.seed + split_index * 100000 + index,
                    )
                )
    elif config.data.name == "manifest":
        records = read_manifest(config.data.manifest)
    else:
        raise ValueError(f"Unknown data source: {config.data.name}")
    identity = {
        "records": records,
        "synthetic_image_size": config.data.get("image_size"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    datasets = {}
    for split in [evaluation_split] if evaluation_split else split_names:
        selected = [
            r
            for r in records
            if r["split"] == split and (not real_only or r["label"] == 0)
        ]
        # Test data may be absent during development; train and validation are required.
        if not selected and split == config.data.test_split and not evaluation_split:
            continue
        views = ImageViews(
            config.views.global_size,
            config.views.local_size,
            config.views.patches,
            config.seed,
            split == config.data.train_split and evaluation_split is None,
        )
        datasets[split] = DetectionDataset(
            selected,
            views,
            config.data.name == "synthetic",
            config.data.get("image_size", 320),
            config.get("augment"),
        )
    return datasets, fingerprint
