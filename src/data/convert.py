"""Pack loose images and ZIP members into WebDataset-compatible tar shards."""

import io
import json
import logging
import os
from pathlib import Path
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


def convert_images(data_dir, output_dir, *, shard_size=10000, temp_dir=None):
    """Convert sequentially, keeping only one decoded image in memory.

    PNG output is lossless, RGB and exactly 128x128. Resize stretches each
    image to the target dimensions without cropping. Animated images use
    their first frame; transparency is composited on white.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"DATA_DIR is not a directory: {data_dir}")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if output_dir == data_dir or data_dir.is_relative_to(output_dir):
        raise ValueError("Output must not be the input directory or its ancestor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    Image.init()
    extensions = set(Image.registered_extensions())
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "failed_images": 0, "failed_archives": 0,
             "incomplete_archives": 0, "ignored_files": 0}
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
            try:
                with Image.open(stream) as original:
                    original_size = list(original.size)
                    oriented = ImageOps.exif_transpose(original)
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    image = background.convert("RGB").resize(
                        (128, 128), Image.Resampling.LANCZOS,
                    )
                    encoded = io.BytesIO()
                    image.save(encoded, format="PNG")
                payload = encoded.getvalue()
                metadata = json.dumps({"source": source, "original_size": original_size,
                                       "size": [128, 128], "mode": "RGB"}).encode()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                report(source, "failed_images", exc)
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

    manifest = {"format": "webdataset", "size": [128, 128], "mode": "RGB",
                "resize": "stretch", "data_dir": str(data_dir), **stats, "shards": shards}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
