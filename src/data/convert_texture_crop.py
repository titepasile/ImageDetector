"""Pack loose images and ZIP members into WebDataset-compatible tar shards.

Each image is reduced to ONE native-resolution 128x128 crop (no resizing).
The crop is drawn from a grid of candidate windows with probability that grows
with the window's high-frequency energy (Laplacian), i.e. soft sampling:
textured regions are preferred, but flat regions keep a non-zero chance.
"""

import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager

from PIL import Image, ImageOps

LOG = logging.getLogger(__name__)


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
                # Recompute overlap boundaries after rebasing local headers.
                end = archive.start_dir
                for member in sorted(archive.infolist(), key=lambda m: m.header_offset, reverse=True):
                    member._end_offset = end
                    end = member.header_offset
                yield archive
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive


def normalize_split_end(stream, offsets):
    """Normalize a temporary ZIP's end records, including ZIP64 records.

    Central directory entries retain their original disk numbers so local
    file headers can be rebased after ZipFile has decoded ZIP64 extra fields.
    """
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


def _grid_positions(extent, size, margin, step):
    """Window start coordinates along one axis, always incl. first and last."""
    last = extent - size - margin
    return sorted(set(range(margin, last + 1, step)) | {last})


def select_texture_crop(gray, size, *, stride, margin, uniform_mix,
                        max_candidates, rng):
    """Pick a size x size window, sampled with probability ~ texture score.

    Texture score = RMS of the Laplacian inside the window (high-frequency
    energy). Window sums come from an integral image, so all candidates are
    scored in O(1) each. Selection probability:

        p_i = (1 - uniform_mix) * score_i / sum(score) + uniform_mix / n

    so every candidate keeps at least uniform_mix / n probability (soft
    sampling, never a hard argmax).

    Returns ((x0, y0, x1, y1), info_dict).
    """
    gray, width, height = gray
    if margin and (width < size + 2 * margin or height < size + 2 * margin):
        margin = 0  # image too small to exclude borders

    step = stride
    while True:
        xs = _grid_positions(width, size, margin, step)
        ys = _grid_positions(height, size, margin, step)
        if len(xs) * len(ys) <= max_candidates or step >= max(width, height):
            break
        step *= 2

    integral = [[0.0] * width for _ in range(height - 1)]
    for y in range(1, height - 1):
        row_sum = 0.0
        for x in range(1, width - 1):
            center = gray[y * width + x]
            laplacian = (gray[(y - 1) * width + x] + gray[(y + 1) * width + x]
                         + gray[y * width + x - 1] + gray[y * width + x + 1]
                         - 4.0 * center)
            row_sum += laplacian * laplacian
            integral[y][x] = integral[y - 1][x] + row_sum

    inner = size - 2  # Laplacian is undefined on the 1px window border
    positions = [(x, y) for y in ys for x in xs]
    scores = []
    for x, y in positions:
        window_sum = (integral[y + inner][x + inner] - integral[y][x + inner]
                      - integral[y + inner][x] + integral[y][x])
        scores.append(math.sqrt(max(window_sum, 0.0) / (inner * inner)))

    n = len(scores)
    total = sum(scores)
    if total > 0:
        probs = [(1.0 - uniform_mix) * score / total + uniform_mix / n
                 for score in scores]
        probability_total = sum(probs)
        probs = [probability / probability_total for probability in probs]
    else:  # completely flat image
        probs = [1.0 / n] * n
    choice = rng.choices(range(n), weights=probs, k=1)[0]
    x0, y0 = positions[choice]
    info = {
        "crop_box": [x0, y0, x0 + size, y0 + size],
        "texture_score": round(float(scores[choice]), 4),
        "mean_candidate_score": round(sum(scores) / n, 4),
        "selection_prob": round(float(probs[choice]), 6),
        "num_candidates": int(n),
    }
    return (x0, y0, x0 + size, y0 + size), info


def convert_images(data_dir, output_dir, *, shard_size=10000, temp_dir=None,
                   crop_size=128, grid_stride=64, border_margin=8,
                   uniform_mix=0.1, max_candidates=1024, seed=0):
    """Convert sequentially, keeping only one decoded image in memory.

    PNG output is lossless, RGB and exactly crop_size x crop_size. Nothing is
    resized: one native-resolution crop is taken per image, chosen by
    texture-weighted soft sampling (see select_texture_crop). Images smaller
    than the crop in either dimension are skipped (never upscaled) and listed
    in errors.jsonl. The RNG is seeded per image from (seed, source), so the
    result is reproducible and independent of processing order. Animated
    images use their first frame; transparency is composited on white.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"DATA_DIR is not a directory: {data_dir}")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if crop_size < 8 or grid_stride < 1 or max_candidates < 4:
        raise ValueError("Invalid crop_size, grid_stride or max_candidates")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    if output_dir == data_dir or data_dir.is_relative_to(output_dir):
        raise ValueError("Output must not be the input directory or its ancestor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    Image.init()
    extensions = set(Image.registered_extensions())
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "failed_images": 0, "failed_archives": 0,
             "incomplete_archives": 0, "skipped_small_images": 0,
             "ignored_files": 0}
    shards = []
    archive_out = None
    errors_path = output_dir / "errors.jsonl"

    with errors_path.open("w", encoding="utf-8") as errors:
        def report(source, kind, exc):
            stats[kind] += 1
            errors.write(json.dumps({"source": source, "error": str(exc)}) + "\n")
            errors.flush()
            LOG.warning("%s: %s", source, exc)

        def add_image(stream, source):
            nonlocal archive_out
            too_small = None
            try:
                with Image.open(stream) as original:
                    original_size = list(original.size)
                    oriented = ImageOps.exif_transpose(original)
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    rgb = background.convert("RGB")
                    width, height = rgb.size
                    if width < crop_size or height < crop_size:
                        too_small = f"image {width}x{height} smaller than {crop_size}x{crop_size}"
                    else:
                        gray_image = rgb.convert("L")
                        gray = (gray_image.getdata(), width, height)
                        digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
                        rng = random.Random(int.from_bytes(digest[:8], "big"))
                        box, info = select_texture_crop(
                            gray, crop_size, stride=grid_stride,
                            margin=border_margin, uniform_mix=uniform_mix,
                            max_candidates=max_candidates, rng=rng)
                        image = rgb.crop(box)
                        encoded = io.BytesIO()
                        image.save(encoded, format="PNG")
                if too_small is None:
                    payload = encoded.getvalue()
                    metadata = json.dumps({
                        "source": source, "original_size": original_size,
                        "size": [crop_size, crop_size], "mode": "RGB",
                        **info}).encode()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                report(source, "failed_images", exc)
                return
            if too_small is not None:
                report(source, "skipped_small_images", too_small)
                return
            if stats["images"] % shard_size == 0:
                if archive_out:
                    archive_out.close()
                    (output_dir / (shards[-1]["name"] + ".partial")).rename(
                        output_dir / shards[-1]["name"])
                name = f"images-{len(shards):06d}.tar"
                archive_out = tarfile.open(output_dir / (name + ".partial"), "w")
                shards.append({"name": name, "images": 0})
            key = f"{stats['images']:012d}"
            for extension, contents in (("png", payload), ("json", metadata)):
                member = tarfile.TarInfo(f"{key}.{extension}")
                member.size = len(contents)
                archive_out.addfile(member, io.BytesIO(contents))
            stats["images"] += 1
            shards[-1]["images"] += 1
            if stats["images"] % 1000 == 0:
                LOG.info("Converted %d images", stats["images"])

        try:
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
                                        with archive.open(member) as stream:
                                            add_image(stream, member_source)
                                    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                                        report(member_source, "failed_images", exc)
                        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                            report(source, "failed_archives", exc)
                    elif suffix == ".part" or re.fullmatch(r"\.z\d+", suffix):
                        if suffix == ".part" or path.stem.lower() not in zip_stems:
                            report(source, "incomplete_archives", "Incomplete download or missing final .zip file")
                    elif suffix in extensions:
                        try:
                            with path.open("rb") as stream:
                                add_image(stream, source)
                        except OSError as exc:
                            report(source, "failed_images", exc)
                    else:
                        stats["ignored_files"] += 1
        finally:
            if archive_out:
                archive_out.close()
        if shards:
            (output_dir / (shards[-1]["name"] + ".partial")).rename(output_dir / shards[-1]["name"])

    manifest = {"format": "webdataset", "size": [crop_size, crop_size], "mode": "RGB",
                "resize": "none",
                "crop": {"method": "texture_weighted_soft_sampling",
                         "score": "laplacian_rms", "grid_stride": grid_stride,
                         "border_margin": border_margin,
                         "uniform_mix": uniform_mix,
                         "max_candidates": max_candidates, "seed": seed},
                "data_dir": str(data_dir), **stats, "shards": shards}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
