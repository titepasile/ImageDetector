"""Save a contact sheet of random crops, one row per generator and class.

Look at it before training. Things to watch for:
  * logos / watermarks / text in the AI crops (a shortcut the network
    would happily learn, and the texture sampling tends to prefer them)
  * borders, padding or completely flat crops
  * obvious colour or brightness differences between real and AI rows

Usage:
    python preview_crops.py SHARDS_DIR preview.png --split train --per-row 12
"""

import argparse
import io
import random
from collections import defaultdict

from PIL import Image, ImageDraw

from local_dataset import iter_shard, list_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards_dir")
    parser.add_argument("output")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-row", type=int, default=12)
    parser.add_argument("--max-shards", type=int, default=20, help="shards to sample from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shards, _ = list_shards(args.shards_dir, args.split)
    rng.shuffle(shards)
    rows = defaultdict(list)  # (generator, label) -> reservoir of PNG bytes
    seen = defaultdict(int)
    for path in shards[:args.max_shards]:
        for png, meta in iter_shard(path):
            key = (meta["generator"], "ai" if meta["label"] == 1 else "real")
            seen[key] += 1
            if len(rows[key]) < args.per_row:
                rows[key].append(png)
            else:
                index = rng.randrange(seen[key])
                if index < args.per_row:
                    rows[key][index] = png

    label_width, size, gap = 260, 128, 4
    keys = sorted(rows)
    sheet = Image.new("RGB", (label_width + args.per_row * (size + gap),
                              len(keys) * (size + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, key in enumerate(keys):
        y = row * (size + gap)
        draw.text((8, y + size // 2 - 6), f"{key[0]} / {key[1]}", fill="black")
        for column, png in enumerate(rows[key]):
            sheet.paste(Image.open(io.BytesIO(png)).convert("RGB"),
                        (label_width + column * (size + gap), y))
    sheet.save(args.output)
    print(f"{len(keys)} rows -> {args.output}")


if __name__ == "__main__":
    main()
