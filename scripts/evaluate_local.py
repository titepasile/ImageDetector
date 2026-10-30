"""Evaluate a trained local encoder on the test split (GenImage 'val').

Reports, at crop level and image level (crops of an image averaged):
  * overall accuracy, accuracy on real and on AI images, AUC
  * the same per generator (e.g. trained on all, how good on Midjourney?)
  * optionally: robustness to extra JPEG compression (--jpeg-qualities).
    The test crops are re-compressed at each quality. If accuracy on AI
    images collapses while real images stay correct, the model has learned
    "JPEG = real" instead of real generator traces.

Usage:
    python evaluate_local.py SHARDS_DIR runs/local_v1/best.pt
    python evaluate_local.py SHARDS_DIR runs/local_v1/best.pt --jpeg-qualities 90 75 50
Writes the results next to the checkpoint as eval_<split>.json.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.local_dataset import Augment, CropShards, list_shards
from src.local_metrics import binary_metrics
from src.model.local_res_net import load_encoder
from src.local_training import evaluate, pick_device


def read_generators(shards_dir):
    """image_id -> generator, from the converter's index.csv."""
    with open(Path(shards_dir) / "index.csv", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["generator"] for row in csv.DictReader(handle)}


def per_generator(collector, generators):
    groups = defaultdict(lambda: ([], []))
    for image_id, (label, score, _) in collector.per_image().items():
        labels, scores = groups[generators.get(image_id, "unknown")]
        labels.append(label)
        scores.append(score)
    return {name: binary_metrics(labels, scores, prefix="image_")
            for name, (labels, scores) in sorted(groups.items())}


def loader_for(shards, batch_size, workers, jpeg_quality=None):
    augment = None
    if jpeg_quality:
        augment = Augment(hflip=False, jpeg_prob=1.0, jpeg_quality=(jpeg_quality, jpeg_quality))
    dataset = CropShards(shards, train=False)
    dataset.augment = augment  # deterministic: fixed quality, no flips
    return DataLoader(dataset, batch_size=batch_size, num_workers=min(workers, len(shards)))


def print_row(name, m):
    def fmt(key):
        return f"{m[key]:.3f}" if key in m else "  -  "
    print(f"  {name:<28} acc {fmt('image_accuracy')}  real {fmt('image_accuracy_real')}  "
          f"ai {fmt('image_accuracy_ai')}  auc {fmt('image_auc')}  n={m.get('image_n', 0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jpeg-qualities", type=int, nargs="*", default=[])
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_encoder(args.checkpoint, map_location=device)
    model.to(device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    shards, _ = list_shards(args.shards_dir, args.split)
    generators = read_generators(args.shards_dir)

    results = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
               "split": args.split}
    print(f"Checkpoint from epoch {checkpoint['epoch']}, split '{args.split}'")

    metrics, collector = evaluate(model, loader_for(shards, args.batch_size, args.workers),
                                  device, amp_dtype, prefix="")
    results["overall"] = metrics
    results["per_generator"] = per_generator(collector, generators)
    print(f"crop level:  acc {metrics['crop_accuracy']:.3f}  auc {metrics.get('crop_auc', float('nan')):.3f}")
    print("image level (crops averaged):")
    print_row("all generators", metrics)
    for name, m in results["per_generator"].items():
        print_row(name, m)

    if args.jpeg_qualities:
        print("extra JPEG compression of the test crops:")
        results["jpeg"] = {}
        for quality in args.jpeg_qualities:
            m, _ = evaluate(model, loader_for(shards, args.batch_size, args.workers, quality),
                            device, amp_dtype, prefix="")
            results["jpeg"][str(quality)] = m
            print_row(f"JPEG quality {quality}", m)

    out_path = Path(args.checkpoint).with_name(f"eval_{args.split}.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
