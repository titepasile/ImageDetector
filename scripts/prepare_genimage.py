"""Create a format-debiased manifest from a GenImage archive.

python -m scripts.prepare_genimage C:/data/ADM/imagenet_ai_0508_adm.zip data/genimage_adm
python -m scripts.prepare_genimage C:/data/BigGAN/imagenet_ai_0419_biggan.zip data/genimage_biggan \
    --train 0 --val 0 --test 1000 --size 128 --allow-missing-volumes
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
    parser.add_argument("--size", type=int, default=256,
                        help="Generator resolution; real images are cropped to it")
    parser.add_argument("--output-size", type=int,
                        help="Resize both classes identically after compression")
    parser.add_argument("--allow-missing-volumes", action="store_true",
                        help="Skip images stored in absent split ZIP volumes")
    args = parser.parse_args()
    counts = {
        split: (source, count)
        for split, source, count in (
            ("train", "train", args.train),
            ("val_id", "train", args.val),
            ("test_id", "val", args.test),
        )
        if count > 0
    }
    summary = build_genimage_manifest(
        args.archive,
        args.output_dir,
        counts,
        args.seed,
        size=args.size,
        output_size=args.output_size,
        allow_missing_volumes=args.allow_missing_volumes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
