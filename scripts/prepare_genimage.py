"""Create a format-debiased manifest from a GenImage archive.

python -m scripts.prepare_genimage C:/data/ADM/imagenet_ai_0508_adm.zip data/genimage_adm
"""

import argparse
import json
from pathlib import Path

from src.data.genimage import build_genimage_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path, help="Final .zip of a (split) GenImage archive")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train", type=int, default=4000, help="Images per class")
    parser.add_argument("--val", type=int, default=1000, help="Images per class, from train/")
    parser.add_argument("--test", type=int, default=1000, help="Images per class, from val/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = {
        "train": ("train", args.train),
        "val_id": ("train", args.val),
        "test_id": ("val", args.test),
    }
    summary = build_genimage_manifest(args.archive, args.output_dir, counts, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
