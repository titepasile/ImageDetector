"""Deterministic test-time perturbations applied identically to both classes.

Each operation is written as "name:argument", for example "jpeg:75". They are
used for robustness tests (jpeg, resample, blur), frequency-band tests
(lowpass, highpass), content tests (shuffle) and border tests (border_gray,
border_reflect). Perturbed copies are stored as PNG with a new manifest.
"""

import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter

from .dataset import read_manifest
from .views import canonical_image

# ImageNet mean colour, also used as letterbox fill in the global view.
MEAN_RGB = (124, 116, 104)


def _frequency_split(image, cutoff):
    pixels = np.asarray(image, dtype=np.float64) / 255
    height, width = pixels.shape[:2]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    mask = (fx**2 + fy**2) <= cutoff**2
    spectrum = np.fft.rfft2(pixels, axes=(0, 1))
    low = np.fft.irfft2(spectrum * mask[..., None], s=(height, width), axes=(0, 1))
    return low, pixels - low


def _to_image(pixels):
    return Image.fromarray(np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8))


def apply_operation(image, operation, seed_text):
    name, _, argument = operation.partition(":")
    width, height = image.size
    if name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(argument))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    if name == "resample":
        # Down- and upscale again: same size, resampling damage only.
        scale = float(argument)
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(float(argument)))
    if name == "lowpass":
        return _to_image(_frequency_split(image, float(argument))[0])
    if name == "highpass":
        return _to_image(_frequency_split(image, float(argument))[1] + 0.5)
    if name == "shuffle":
        tile = int(argument)
        if width % tile or height % tile:
            raise ValueError(f"Image size {image.size} is not a multiple of {tile}")
        pixels = np.asarray(image)
        tiles = [
            pixels[y : y + tile, x : x + tile]
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8]))
        rng.shuffle(tiles)
        columns = width // tile
        rows = [np.concatenate(tiles[i : i + columns], axis=1) for i in range(0, len(tiles), columns)]
        return Image.fromarray(np.concatenate(rows, axis=0))
    if name == "border_gray":
        border = int(argument)
        result = Image.new("RGB", image.size, MEAN_RGB)
        inner = (border, border, width - border, height - border)
        result.paste(image.crop(inner), inner[:2])
        return result
    if name == "border_reflect":
        border = int(argument)
        pixels = np.asarray(image)[border : height - border, border : width - border]
        padded = np.pad(pixels, ((border, border), (border, border), (0, 0)), mode="reflect")
        return Image.fromarray(padded)
    if name == "crop":
        size = int(argument)
        x = ((width - size) // 2) // 8 * 8
        y = ((height - size) // 2) // 8 * 8
        return image.crop((x, y, x + size, y + size))
    raise ValueError(f"Unknown perturbation: {operation}")


def perturb_manifest(manifest, split, output_dir, operations):
    """Write perturbed copies of one split ("all" for every split) and a manifest."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    records = [r for r in read_manifest(manifest) if split == "all" or r["split"] == split]
    if not records:
        raise ValueError(f"Split {split!r} is empty in {manifest}")
    lines = []
    for index, record in enumerate(records):
        with Image.open(record["path"]) as original:
            image = canonical_image(original)
        for operation in operations:
            image = apply_operation(image, operation, f"{record['sample_id']}:{operation}")
        kind = "ai" if record["label"] else "nature"
        relative = Path(record["split"]) / kind / f"{index:06d}.png"
        (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / relative, "PNG")
        new = {k: v for k, v in record.items() if k not in ("path", "sha256", "sample_id")}
        new.update(
            path=relative.as_posix(),
            source_path=record["path"],
            perturbation=list(operations),
        )
        lines.append(json.dumps(new))
    (output_dir / "samples.jsonl").write_text("\n".join(lines) + "\n")
    return len(lines)
