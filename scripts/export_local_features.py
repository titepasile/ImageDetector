"""Write the local features z2 of every image, for stage B.

For each split, OUT/local_features_<split>.npz contains
    image_id   (N,)     same id as in index.csv, used to join with z1
    source     (N,)     original path, a second way to join
    label      (N,)     0 = real, 1 = AI
    generator  (N,)
    z2         (N, D)   mean of the crop features of the image
    num_crops  (N,)

The fc head is NOT used here, only model.features(). The crops are not
augmented, so the export is deterministic.

How stage B uses it (whatever head you choose later):
    local = np.load("local_features_train.npz")
    z = np.concatenate([z1_of_same_images, local["z2"]], axis=1)

Joining with the global branch works on image_id as long as both converters
were run on the same DATA_DIR (so that 'source' strings are identical).

Usage:
    python export_local_features.py SHARDS_DIR runs/local_v1/best.pt features/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.local_dataset import CropShards, list_shards
from src.model.local_res_net import load_encoder
from src.local_training import pick_device


@torch.no_grad()
def export_split(model, shards, device, batch_size, workers):
    sums = {}
    counts = defaultdict(int)
    labels = {}
    loader = DataLoader(CropShards(shards, train=False), batch_size=batch_size,
                        num_workers=min(workers, len(shards)))
    model.eval()
    for images, batch_labels, _, image_ids in loader:
        features = model.features(images.to(device)).float().cpu().numpy()
        for image_id, label, feature in zip(image_ids, batch_labels.tolist(), features):
            if image_id in sums:
                sums[image_id] += feature
            else:
                sums[image_id] = feature.astype(np.float64)
            counts[image_id] += 1
            labels[image_id] = label
    ids = sorted(sums)
    z2 = np.stack([sums[i] / counts[i] for i in ids]).astype(np.float32)
    return ids, z2, [labels[i] for i in ids], [counts[i] for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_encoder(args.checkpoint, map_location=device)
    model.to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(args.shards_dir) / "index.csv", newline="", encoding="utf-8") as handle:
        index = {row["image_id"]: row for row in csv.DictReader(handle)}

    for split in args.splits:
        shards, _ = list_shards(args.shards_dir, split)
        ids, z2, labels, counts = export_split(model, shards, device, args.batch_size, args.workers)
        path = out_dir / f"local_features_{split}.npz"
        np.savez_compressed(
            path, image_id=np.array(ids), z2=z2, label=np.array(labels, dtype=np.int64),
            num_crops=np.array(counts, dtype=np.int64),
            source=np.array([index[i]["source"] for i in ids]),
            generator=np.array([index[i]["generator"] for i in ids]))
        print(f"{split}: {len(ids)} images, z2 shape {z2.shape} -> {path}")


if __name__ == "__main__":
    main()
