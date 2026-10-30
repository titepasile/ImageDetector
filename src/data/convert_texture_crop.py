"""Pack GenImage images (loose or in ZIPs) into tar shards of 128x128 crops.

Changes compared to the first version:
  * Texture scoring uses NumPy (same scores, ~20x faster, far less memory).
  * Several crops per image (default 4), drawn WITHOUT replacement with
    probability that grows with the window's Laplacian energy (soft sampling).
  * Label (real/ai), generator and split are read from the GenImage folder
    names and stored in every sample's JSON (see splits.py).
  * Separate shard folders per split: OUT/train, OUT/val, OUT/test.
  * Each image goes to one of several open shards at random, so a shard
    mixes classes and generators instead of holding one folder in order.
  * Optional JPEG harmonisation (default quality 96, applied to ALL images
    before cropping) against the known GenImage JPEG bias: real ImageNet
    photos are JPEG, generated images are PNG. Use --jpeg-quality 0 to turn
    it off. Whatever you choose, the global branch should do the same.
  * index.csv lists every image with id, label, generator and split. It is
    the shared split file for the global branch and for stage B.
  * Decoding and cropping run in worker processes (--workers).

Nothing is ever resized. Images smaller than the crop are skipped.

Usage:
    python convert_texture_crop.py DATA_DIR OUT_DIR --num-crops 4 --workers 8
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import re
import shutil
import struct
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageOps

from splits import assign_split, image_id, parse_source

LOG = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# ZIP handling (unchanged from the first version)
# --------------------------------------------------------------------------

@contextmanager
def readable_zip(path, temp_dir=None):
    """Join split volumes and rebase their disk-relative ZIP offsets."""
    parts = sorted(
        (p for p in path.parent.iterdir()
         if p.stem.lower() == path.stem.lower()
         and re.fullmatch(r"\.z\d+", p.suffix.lower())),
        key=lambda p: int(p.suffix[2:]),
    )
    if parts:
        numbers = [int(p.suffix[2:]) for p in parts]
        if numbers != list(range(1, numbers[-1] + 1)):
            raise ValueError(f"Missing split ZIP volumes for {path}")
        with tempfile.TemporaryDirectory(dir=temp_dir) as temporary:
            joined = Path(temporary) / "joined.zip"
            offsets = []
            with joined.open("w+b") as target:
                for volume in [*parts, path]:
                    offsets.append(target.tell())
                    with volume.open("rb") as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                try:
                    central_offset = normalize_split_end(target, offsets)
                except (struct.error, IndexError) as exc:
                    raise ValueError("Truncated or invalid split ZIP end records") from exc
            with zipfile.ZipFile(joined) as archive:
                # ZipFile normally adds a prefix correction for concatenated
                # archives. End records were normalized, so that correction is 0.
                if archive.start_dir != central_offset:
                    raise ValueError("Unexpected central directory offset in split ZIP")
                for member in archive.infolist():
                    if member.volume >= len(offsets):
                        raise ValueError("ZIP references a missing volume")
                    member.header_offset += offsets[member.volume]
                # Recompute overlap boundaries after rebasing local headers
                # (private attribute, only used by Python >= 3.12).
                end = archive.start_dir
                for member in sorted(archive.infolist(), key=lambda m: m.header_offset, reverse=True):
                    member._end_offset = end
                    end = member.header_offset
                yield archive
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive


def normalize_split_end(stream, offsets):
    """Normalize a temporary ZIP's end records, including ZIP64 records."""
    stream.seek(0, 2)
    size = stream.tell()
    tail_start = max(0, size - 65557)
    stream.seek(tail_start)
    tail = stream.read()
    index = tail.rfind(b"PK\x05\x06")
    # A ZIP comment may itself contain the signature.
    while index >= 0:
        if index + 22 <= len(tail):
            end = list(struct.unpack_from("<4s4H2IH", tail, index))
            if index + 22 + end[-1] == len(tail):
                break
        index = tail.rfind(b"PK\x05\x06", 0, index)
    else:
        raise ValueError("Split ZIP has no complete end record")
    position = tail_start + index
    stream.seek(max(0, position - 20))
    locator_bytes = stream.read(20)
    if locator_bytes[:4] == b"PK\x06\x07":
        locator = list(struct.unpack("<4sIQI", locator_bytes))
        if locator[1] >= len(offsets) or locator[3] != len(offsets):
            raise ValueError("ZIP64 references missing volumes")
        zip64_position = offsets[locator[1]] + locator[2]
        stream.seek(zip64_position)
        record = list(struct.unpack("<4sQ2H2I4Q", stream.read(56)))
        if record[0] != b"PK\x06\x06" or record[5] >= len(offsets):
            raise ValueError("Invalid split ZIP64 end record")
        central_offset = offsets[record[5]] + record[9]
        record[4:6] = [0, 0]
        record[6] = record[7]
        record[9] = central_offset
        stream.seek(zip64_position)
        stream.write(struct.pack("<4sQ2H2I4Q", *record))
        locator[1:] = [0, zip64_position, 1]
        stream.seek(position - 20)
        stream.write(struct.pack("<4sIQI", *locator))
    else:
        if end[1] != len(offsets) - 1 or end[2] >= len(offsets):
            raise ValueError("Split ZIP references missing volumes")
        central_offset = offsets[end[2]] + end[6]
    end[1:3] = [0, 0]
    end[3] = end[4]
    end[6] = min(central_offset, 0xFFFFFFFF)
    stream.seek(position)
    stream.write(struct.pack("<4s4H2IH", *end))
    return central_offset


# --------------------------------------------------------------------------
# Texture-weighted crop selection (NumPy)
# --------------------------------------------------------------------------

def _grid_positions(extent, size, margin, step):
    """Window start coordinates along one axis, always incl. first and last."""
    last = extent - size - margin
    return sorted(set(range(margin, last + 1, step)) | {last})


def laplacian_integral(gray):
    """Integral image of the squared 4-neighbour Laplacian.

    Same convention as the first version: ii[y, x] = sum of laplacian^2 over
    rows 1..y and columns 1..x (row/column 0 and the last row/column are 0,
    because the Laplacian is undefined on the 1px image border).
    """
    g = np.asarray(gray, dtype=np.float32)
    height, width = g.shape
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4.0 * g[1:-1, 1:-1]).astype(np.float64)
    np.square(lap, out=lap)
    ii = np.zeros((height, width), dtype=np.float64)
    ii[1:height - 1, 1:width - 1] = lap.cumsum(axis=0).cumsum(axis=1)
    return ii


def select_texture_crops(gray, size, *, num_crops, stride, margin, uniform_mix,
                         max_candidates, rng):
    """Pick up to num_crops distinct size x size windows, texture-weighted.

    Texture score = RMS of the Laplacian inside the window. Probabilities:

        p_i = (1 - uniform_mix) * score_i / sum(score) + uniform_mix / n

    Windows are drawn without replacement (weighted reservoir sampling), so
    the crops of one image never repeat. Fewer than num_crops are returned
    only if the image has fewer candidate windows.

    gray: 2-D uint8 array (height, width). rng: numpy Generator.
    Returns a list of ((x0, y0, x1, y1), info_dict).
    """
    height, width = gray.shape
    if margin and (width < size + 2 * margin or height < size + 2 * margin):
        margin = 0  # image too small to exclude borders

    step = stride
    while True:
        xs = _grid_positions(width, size, margin, step)
        ys = _grid_positions(height, size, margin, step)
        if len(xs) * len(ys) <= max_candidates or step >= max(width, height):
            break
        step *= 2

    ii = laplacian_integral(gray)
    inner = size - 2
    x = np.array(xs)[None, :]
    y = np.array(ys)[:, None]
    window = (ii[y + inner, x + inner] - ii[y, x + inner]
              - ii[y + inner, x] + ii[y, x])
    scores = np.sqrt(np.maximum(window, 0.0) / (inner * inner)).ravel()
    positions = [(px, py) for py in ys for px in xs]

    n = len(scores)
    total = scores.sum()
    if total > 0:
        probs = (1.0 - uniform_mix) * scores / total + uniform_mix / n
        probs = probs / probs.sum()
    else:  # completely flat image
        probs = np.full(n, 1.0 / n)

    k = min(num_crops, n)
    # Efraimidis-Spirakis: the k largest u^(1/p) are a weighted sample without
    # replacement. Done in log space for numerical stability.
    keys = np.log(rng.random(n)) / probs
    chosen = np.argsort(-keys)[:k]

    results = []
    for index in chosen:
        x0, y0 = positions[index]
        results.append(((x0, y0, x0 + size, y0 + size), {
            "crop_box": [int(x0), int(y0), int(x0 + size), int(y0 + size)],
            "texture_score": round(float(scores[index]), 4),
            "mean_candidate_score": round(float(scores.mean()), 4),
            "selection_prob": round(float(probs[index]), 6),
            "num_candidates": int(n),
        }))
    return results


# --------------------------------------------------------------------------
# Per-image work (runs in worker processes)
# --------------------------------------------------------------------------

def process_image(source, data, config):
    """Decode one image and return its crops, or an error description.

    Returns {"status": "ok", ...} or {"status": <error kind>, "error": str}.
    """
    parsed = parse_source(source)
    if parsed is None:
        return {"status": "unlabeled_images",
                "error": "path has no train|val/ai|nature folders"}
    label, generator, genimage_split = parsed
    crop_size = config["crop_size"]
    try:
        with Image.open(io.BytesIO(data)) as original:
            original_format = original.format or "unknown"
            original_size = list(original.size)
            oriented = ImageOps.exif_transpose(original)
            rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        rgb = background.convert("RGB")
        width, height = rgb.size
        if width < crop_size or height < crop_size:
            return {"status": "skipped_small_images",
                    "error": f"image {width}x{height} smaller than {crop_size}x{crop_size}"}
        if config["jpeg_quality"]:
            # Same JPEG treatment for real and generated images, on the full
            # image (like a real camera/website would), before cropping.
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=config["jpeg_quality"])
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                rgb = compressed.convert("RGB")
        digest = hashlib.sha256(f"{config['seed']}:{source}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        crops = select_texture_crops(
            np.asarray(rgb.convert("L")), crop_size,
            num_crops=config["num_crops"], stride=config["grid_stride"],
            margin=config["border_margin"], uniform_mix=config["uniform_mix"],
            max_candidates=config["max_candidates"], rng=rng)
        samples = []
        for crop_index, (box, info) in enumerate(crops):
            encoded = io.BytesIO()
            rgb.crop(box).save(encoded, format="PNG")
            samples.append((crop_index, encoded.getvalue(), info))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return {"status": "failed_images", "error": str(exc)}
    return {
        "status": "ok", "label": label, "generator": generator,
        "split": assign_split(source, genimage_split, config["val_fraction"]),
        "original_format": original_format, "original_size": original_size,
        "samples": samples,
    }


# --------------------------------------------------------------------------
# Shard writing
# --------------------------------------------------------------------------

class ShardPool:
    """A few open tar shards per split; each image goes to a random one.

    Writing images into one shard in folder order would put e.g. 5000
    Midjourney-AI crops in a row into one file. Spreading them over several
    open shards mixes classes and generators inside every shard.
    """

    def __init__(self, root, split, shard_size, open_shards, rng):
        self.directory = root / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.rng = rng
        self.slots = [None] * open_shards  # [tarfile, name, count]
        self.shards = []
        self.next_index = 0

    def _open(self, slot):
        name = f"{self.split}-{self.next_index:06d}.tar"
        self.next_index += 1
        handle = tarfile.open(self.directory / (name + ".partial"), "w")
        self.slots[slot] = [handle, name, 0]

    def _close(self, slot):
        handle, name, count = self.slots[slot]
        handle.close()
        (self.directory / (name + ".partial")).rename(self.directory / name)
        self.shards.append({"name": f"{self.split}/{name}", "samples": count})
        self.slots[slot] = None

    def write_image(self, members):
        """members: list of (filename, bytes). All go into the same shard."""
        slot = self.rng.randrange(len(self.slots))
        if self.slots[slot] is None:
            self._open(slot)
        handle = self.slots[slot][0]
        for filename, contents in members:
            info = tarfile.TarInfo(filename)
            info.size = len(contents)
            handle.addfile(info, io.BytesIO(contents))
        self.slots[slot][2] += sum(1 for f, _ in members if f.endswith(".png"))
        if self.slots[slot][2] >= self.shard_size:
            self._close(slot)

    def close(self):
        for slot, entry in enumerate(self.slots):
            if entry is not None:
                self._close(slot)


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------

def iter_inputs(data_dir, output_dir, extensions, temp_dir, report):
    """Yield (source, bytes) for every image file and ZIP member, in order."""
    for directory, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs
                         if not (Path(directory) / d).resolve().is_relative_to(output_dir))
        zip_stems = {Path(f).stem.lower() for f in files
                     if Path(f).suffix.lower() == ".zip"}
        for filename in sorted(files):
            path = Path(directory) / filename
            source = path.relative_to(data_dir).as_posix()
            suffix = path.suffix.lower()
            if suffix == ".zip":
                try:
                    with readable_zip(path, temp_dir) as archive:
                        for member in archive.infolist():
                            if member.is_dir() or Path(member.filename).suffix.lower() not in extensions:
                                continue
                            member_source = f"{source}!{member.filename}"
                            try:
                                data = archive.read(member)
                            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                                report(member_source, "failed_images", exc)
                                continue
                            yield member_source, data
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                    report(source, "failed_archives", exc)
            elif suffix == ".part" or re.fullmatch(r"\.z\d+", suffix):
                if suffix == ".part" or path.stem.lower() not in zip_stems:
                    report(source, "incomplete_archives",
                           "Incomplete download or missing final .zip file")
            elif suffix in extensions:
                try:
                    yield source, path.read_bytes()
                except OSError as exc:
                    report(source, "failed_images", exc)
            else:
                report(source, "ignored_files", "not an image or archive")


def convert_images(data_dir, output_dir, *, shard_size=5000, open_shards=16,
                   temp_dir=None, crop_size=128, num_crops=4, grid_stride=64,
                   border_margin=8, uniform_mix=0.1, max_candidates=1024,
                   jpeg_quality=96, val_fraction=0.05, seed=0, workers=None):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"DATA_DIR is not a directory: {data_dir}")
    if shard_size < 1 or open_shards < 1 or num_crops < 1:
        raise ValueError("shard_size, open_shards and num_crops must be positive")
    if crop_size < 8 or grid_stride < 1 or max_candidates < 4:
        raise ValueError("Invalid crop_size, grid_stride or max_candidates")
    if not 0.0 <= uniform_mix <= 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("uniform_mix must be in [0, 1], val_fraction in [0, 1)")
    if jpeg_quality and not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be 1..100 or 0 to disable")
    if output_dir == data_dir or data_dir.is_relative_to(output_dir):
        raise ValueError("Output must not be the input directory or its ancestor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    Image.init()
    extensions = set(Image.registered_extensions())
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    config = {"crop_size": crop_size, "num_crops": num_crops,
              "grid_stride": grid_stride, "border_margin": border_margin,
              "uniform_mix": uniform_mix, "max_candidates": max_candidates,
              "jpeg_quality": jpeg_quality, "val_fraction": val_fraction,
              "seed": seed}
    stats = {"images": 0, "crops": 0, "failed_images": 0, "failed_archives": 0,
             "incomplete_archives": 0, "skipped_small_images": 0,
             "unlabeled_images": 0, "ignored_files": 0}
    per_split = {split: {"images": 0, "crops": 0, "real": 0, "ai": 0} for split in SPLITS}
    generators = {}
    formats = {}
    shard_rng = random.Random(seed)
    pools = {split: ShardPool(output_dir, split, shard_size, open_shards, shard_rng)
             for split in SPLITS}

    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as errors, \
         (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as index_file:
        index = csv.writer(index_file)
        index.writerow(["image_id", "source", "label", "generator", "split",
                        "original_format", "width", "height", "num_crops"])

        def report(source, kind, exc):
            stats[kind] += 1
            errors.write(json.dumps({"source": source, "kind": kind, "error": str(exc)}) + "\n")
            errors.flush()
            if kind != "ignored_files":
                LOG.warning("%s: %s", source, exc)

        def handle(source, result):
            if result["status"] != "ok":
                report(source, result["status"], result["error"])
                return
            uid = image_id(source)
            split, label = result["split"], result["label"]
            members = []
            for crop_index, png, info in result["samples"]:
                key = f"{uid}_{crop_index}"
                metadata = {"image_id": uid, "source": source, "label": label,
                            "generator": result["generator"], "split": split,
                            "crop_index": crop_index,
                            "num_crops_of_image": len(result["samples"]),
                            "original_format": result["original_format"],
                            "original_size": result["original_size"],
                            "size": [crop_size, crop_size], "mode": "RGB",
                            "jpeg_quality": jpeg_quality or None, **info}
                members.append((f"{key}.png", png))
                members.append((f"{key}.json", json.dumps(metadata).encode()))
            pools[split].write_image(members)
            n = len(result["samples"])
            stats["images"] += 1
            stats["crops"] += n
            per_split[split]["images"] += 1
            per_split[split]["crops"] += n
            per_split[split]["ai" if label == 1 else "real"] += 1
            generators.setdefault(result["generator"], {"real": 0, "ai": 0})[
                "ai" if label == 1 else "real"] += 1
            fmt_key = f"{'ai' if label == 1 else 'real'}:{result['original_format']}"
            formats[fmt_key] = formats.get(fmt_key, 0) + 1
            index.writerow([uid, source, label, result["generator"], split,
                            result["original_format"], *result["original_size"], n])
            if stats["images"] % 1000 == 0:
                LOG.info("Converted %d images (%d crops)", stats["images"], stats["crops"])
                index_file.flush()

        try:
            inputs = iter_inputs(data_dir, output_dir, extensions, temp_dir, report)
            if workers == 1:
                for source, data in inputs:
                    handle(source, process_image(source, data, config))
            else:
                # Bounded number of images in flight, results in input order.
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    pending = deque()
                    for source, data in inputs:
                        pending.append((source, executor.submit(process_image, source, data, config)))
                        while len(pending) >= workers * 4:
                            done_source, future = pending.popleft()
                            handle(done_source, future.result())
                    while pending:
                        done_source, future = pending.popleft()
                        handle(done_source, future.result())
        finally:
            for pool in pools.values():
                pool.close()

    manifest = {
        "format": "webdataset", "size": [crop_size, crop_size], "mode": "RGB",
        "resize": "none", "labels": {"0": "nature (real)", "1": "ai"},
        "crop": {"method": "texture_weighted_soft_sampling_without_replacement",
                 "score": "laplacian_rms", "num_crops": num_crops,
                 "grid_stride": grid_stride, "border_margin": border_margin,
                 "uniform_mix": uniform_mix, "max_candidates": max_candidates,
                 "seed": seed},
        "jpeg_quality": jpeg_quality or None, "val_fraction": val_fraction,
        "data_dir": str(data_dir), **stats, "splits": per_split,
        "generators": generators, "original_formats": formats,
        "shards": [shard for split in SPLITS for shard in pools[split].shards],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--num-crops", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--jpeg-quality", type=int, default=96,
                        help="re-encode every image as JPEG first; 0 disables")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--shard-size", type=int, default=5000, help="crops per shard")
    parser.add_argument("--open-shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--temp-dir", default=None, help="where split ZIPs are joined")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = convert_images(
        args.data_dir, args.output_dir, num_crops=args.num_crops,
        crop_size=args.crop_size, jpeg_quality=args.jpeg_quality,
        val_fraction=args.val_fraction, shard_size=args.shard_size,
        open_shards=args.open_shards, workers=args.workers,
        temp_dir=args.temp_dir, seed=args.seed)
    LOG.info("Done: %d images, %d crops. Per split: %s",
             manifest["images"], manifest["crops"], manifest["splits"])


if __name__ == "__main__":
    main()
