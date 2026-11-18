"""Build a format-debiased manifest from a GenImage subset (e.g. imagenet_ai_0508_adm).

GenImage stores generated images as 256x256 PNG and real ImageNet images as JPEG
of varying size. A detector can separate these by file format, JPEG artifacts,
image size or aspect ratio alone (Grommelt et al., 2024, "Fake or JPEG?").
This builder removes those cues:

* Real images: decoded once and center-cropped to 256x256, with an offset that
  keeps the original 8x8 JPEG grid aligned to the crop origin. Grayscale,
  non-baseline-subsampled and too-small images are skipped.
* Generated images: JPEG-compressed once with a (quality, subsampling) pair
  drawn from the real images selected for the same split, then decoded.
* Both classes are stored as lossless PNG, so every image is 256x256 RGB with
  exactly one JPEG compression from a matched distribution.
"""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image, JpegImagePlugin

from .spanned_zip import MissingVolumeError, open_spanned_zip
from .views import canonical_image

SIZE = 256
# Standard IJG luminance table used to estimate the JPEG quality setting.
IJG_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]  # fmt: skip
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def estimate_quality(image):
    table = sorted(image.quantization[0])
    best = None
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        expected = sorted(
            min(255, max(1, int((value * scale + 50) // 100))) for value in IJG_LUMA
        )
        distance = sum(abs(a - b) for a, b in zip(expected, table))
        if best is None or distance < best[0]:
            best = (distance, quality)
    return best[1]


def real_sample(data, size=SIZE):
    """Return (size x size RGB crop, jpeg settings) or None if the image is unusable."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            return None
        sampling = JpegImagePlugin.get_sampling(image)
        if sampling not in SUBSAMPLING or min(image.size) < size:
            return None
        settings = {"quality": estimate_quality(image), "subsampling": sampling}
        image.load()
        width, height = image.size
        # Offsets are multiples of 8 so JPEG blocks start at the crop origin.
        x = ((width - size) // 2) // 8 * 8
        y = ((height - size) // 2) // 8 * 8
        return image.crop((x, y, x + size, y + size)).copy(), settings


def generated_sample(data, settings, size=SIZE):
    with Image.open(io.BytesIO(data)) as image:
        image = canonical_image(image)
    if image.size != (size, size):
        return None
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", **settings)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def build_genimage_manifest(
    archive_path,
    output_dir,
    counts,
    seed=0,
    family=None,
    size=SIZE,
    output_size=None,
    allow_missing_volumes=False,
):
    """counts: {split: (source_folder, images_per_class)}; source_folder is train/val.

    Splits drawn from the same source folder never share an image. `size` is the
    generator's native resolution and the real-image crop size. `output_size`
    optionally resizes both classes identically (bicubic) after compression.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    rng = random.Random(seed)
    records, stats = [], Counter()
    with open_spanned_zip(archive_path, allow_missing_volumes) as archive:
        members = {}
        for info in archive.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 4:
                continue
            root, source, kind, _ = parts
            members.setdefault((source, kind), []).append(info)
            family = family or root.split("_")[-1]
        for value in members.values():
            value.sort(key=lambda info: info.filename)
            rng.shuffle(value)
        cursors = Counter()
        for split, (source, per_class) in counts.items():
            settings_pool = []
            for kind, label in (("nature", 0), ("ai", 1)):
                pool = members[source, kind]
                written = 0
                while written < per_class:
                    if cursors[source, kind] >= len(pool):
                        raise ValueError(f"Not enough usable {source}/{kind} images")
                    info = pool[cursors[source, kind]]
                    cursors[source, kind] += 1
                    try:
                        data = archive.read(info)
                    except MissingVolumeError:
                        stats[f"missing_volume_{split}_{kind}"] += 1
                        continue
                    if label == 0:
                        result = real_sample(data, size)
                        if result is None:
                            stats[f"skipped_{split}_real"] += 1
                            continue
                        image, settings = result
                        settings_pool.append(settings)
                    else:
                        settings = rng.choice(settings_pool)
                        image = generated_sample(data, settings, size)
                        if image is None:
                            stats[f"skipped_{split}_generated"] += 1
                            continue
                    if output_size and output_size != size:
                        image = image.resize(
                            (output_size, output_size), Image.Resampling.BICUBIC
                        )
                    stem = Path(info.filename).stem
                    relative = Path(split) / kind / f"{stem}.png"
                    (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
                    image.save(output_dir / relative, "PNG")
                    record = {
                        "path": relative.as_posix(),
                        "label": label,
                        "split": split,
                        "group_id": f"{kind}-{stem}",
                        "source_domain": "imagenet" if label == 0 else family,
                        "source_member": info.filename,
                        "jpeg_quality": settings["quality"],
                        "jpeg_subsampling": SUBSAMPLING[settings["subsampling"]],
                    }
                    if label:
                        record["generator_family"] = family
                    records.append(record)
                    written += 1
                    stats[f"{split}_{kind}"] += 1
    manifest = output_dir / "samples.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "archive": str(archive_path),
        "seed": seed,
        "counts": {k: list(v) for k, v in counts.items()},
        "size": size,
        "output_size": output_size or size,
        "stats": dict(stats),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (output_dir / "build.json").write_text(json.dumps(summary, indent=2))
    return summary
