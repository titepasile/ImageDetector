"""Run with: python -m scripts.convert_images"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data.convert import convert_images

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Convert DATA_DIR images to 128x128 RGB tar shards")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--data-dir", type=Path, help="Override DATA_DIR from .env")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--shard-size", type=int, default=10000, help="Images per tar shard")
    parser.add_argument("--temp-dir", type=Path, help="Temporary storage for joining split ZIPs")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    configured = args.data_dir or os.getenv("DATA_DIR") or os.getenv("DATA_DIr")
    if not configured:
        parser.error("Set DATA_DIR in .env or pass --data-dir")
    data_dir = Path(configured).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest = convert_images(data_dir, args.output_dir, shard_size=args.shard_size,
                                  temp_dir=args.temp_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    logging.info("Converted %d images into %d shards at %s", manifest["images"],
                 len(manifest["shards"]), args.output_dir)
    logging.info("See manifest.json and errors.jsonl for conversion details")
    if not manifest["images"]:
        parser.exit(1, "No images converted. Check errors.jsonl and complete archive downloads.\n")


if __name__ == "__main__":
    main()
