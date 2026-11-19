"""Model and data diagnostics: residual spectra and occlusion sensitivity maps."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
import torch  # noqa: E402

from .data.dataset import read_manifest  # noqa: E402
from .data.perturb import MEAN_RGB  # noqa: E402
from .data.views import ImageViews, canonical_image  # noqa: E402
from .prediction import load_detector  # noqa: E402


def _select(manifest, split, per_class, seed=0):
    records = [r for r in read_manifest(manifest) if r["split"] == split]
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        subset = [r for r in records if r["label"] == label]
        indices = rng.choice(len(subset), min(per_class, len(subset)), replace=False)
        chosen.extend(subset[i] for i in sorted(indices))
    return chosen


def residual_spectrum(image):
    """Mean power spectrum of the grayscale noise residual (image minus 3x3 median)."""
    gray = image.convert("L")
    residual = np.asarray(gray, np.float64) - np.asarray(
        gray.filter(ImageFilter.MedianFilter(3)), np.float64
    )
    residual -= residual.mean()
    return np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2


def spectra(datasets, output_dir, per_class=500):
    """datasets: {name: (manifest, split)}. Saves spectra.png and spectra.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, figure_rows = {}, []
    for name, (manifest, split) in datasets.items():
        sums = {}
        for record in _select(manifest, split, per_class):
            with Image.open(record["path"]) as original:
                power = residual_spectrum(canonical_image(original))
            label = "ai" if record["label"] else "real"
            sums.setdefault(label, []).append(power)
        mean = {label: np.mean(values, axis=0) for label, values in sums.items()}
        # The zero-frequency bin is ~0 after removing the mean; eps avoids NaN there.
        ratio = np.log10(mean["ai"] + 1e-12) - np.log10(mean["real"] + 1e-12)
        ratio[ratio.shape[0] // 2, ratio.shape[1] // 2] = np.median(ratio)
        size = ratio.shape[0]
        # Peak-to-median ratio of the AI/real log ratio: high values mean
        # isolated periodic peaks (e.g. upsampling grids).
        centre = np.ones_like(ratio, dtype=bool)
        centre[size // 2 - 2 : size // 2 + 3, size // 2 - 2 : size // 2 + 3] = False
        results[name] = {
            "images_per_class": {k: len(v) for k, v in sums.items()},
            "size": size,
            "log_ratio_mean": float(ratio[centre].mean()),
            "log_ratio_max": float(ratio[centre].max()),
            "log_ratio_peak_over_median": float(ratio[centre].max() - np.median(ratio[centre])),
        }
        np.savez_compressed(output_dir / f"{name}.npz", **mean, log_ratio=ratio)
        figure_rows.append((name, mean, ratio))
    figure, axes = plt.subplots(len(figure_rows), 3, figsize=(11, 3.6 * len(figure_rows)))
    axes = np.atleast_2d(axes)
    for row, (name, mean, ratio) in zip(axes, figure_rows):
        # Robust colour limits: the zero-frequency bin (removed mean) is extreme.
        for axis, (title, values) in zip(
            row[:2], (("real", np.log10(mean["real"] + 1e-12)), ("ai", np.log10(mean["ai"] + 1e-12)))
        ):
            low, high = np.percentile(values, [1, 99.9])
            image = axis.imshow(values, cmap="magma", vmin=low, vmax=high)
            axis.set_title(f"{name}: {title} (log power)")
            figure.colorbar(image, ax=axis, fraction=0.046)
        centred = ratio - np.median(ratio)
        limit = np.percentile(np.abs(centred), 99.5)
        image = row[2].imshow(centred, cmap="coolwarm", vmin=-limit, vmax=limit)
        row[2].set_title(f"{name}: log10 AI / real (minus median)")
        figure.colorbar(image, ax=row[2], fraction=0.046)
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_dir / "spectra.png", dpi=110)
    plt.close(figure)
    (output_dir / "spectra.json").write_text(json.dumps(results, indent=2))
    return results


@torch.inference_mode()
def occlusion(checkpoint, manifest, split, output_dir, per_class=100, grid=8, device="cpu"):
    """Score change when one of grid x grid cells is filled with the mean colour.

    Positive values: hiding the cell lowers the 'generated' score (evidence for AI).
    Saves occlusion.png and occlusion.json with a border-versus-interior ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config, _ = load_detector(checkpoint, device)
    views = ImageViews(
        config.views.global_size, config.views.local_size, config.views.patches, config.seed
    )
    maps = {"real": [], "ai": []}
    for record in _select(manifest, split, per_class):
        with Image.open(record["path"]) as original:
            image = canonical_image(original)
        width, height = image.size
        cell_w, cell_h = width // grid, height // grid
        variants = [image]
        for row in range(grid):
            for column in range(grid):
                occluded = image.copy()
                box = (column * cell_w, row * cell_h, (column + 1) * cell_w, (row + 1) * cell_h)
                occluded.paste(MEAN_RGB, box)
                variants.append(occluded)
        items = [views(v, record["sample_id"]) for v in variants]
        batch = {k: torch.stack([item[k] for item in items]).to(device) for k in items[0]}
        scores = model(batch)["logits"].softmax(dim=1)[:, 1].cpu().numpy()
        delta = (scores[0] - scores[1:]).reshape(grid, grid)
        maps["ai" if record["label"] else "real"].append(delta)
    stats = {}
    border = np.zeros((grid, grid), dtype=bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, (label, values) in zip(axes, maps.items()):
        mean = np.mean(values, axis=0)
        magnitude = np.mean(np.abs(values), axis=0)
        stats[label] = {
            "images": len(values),
            "mean_signed": mean.tolist(),
            "mean_abs": magnitude.tolist(),
            "border_over_interior_abs": float(magnitude[border].mean() / max(magnitude[~border].mean(), 1e-12)),
        }
        limit = np.abs(mean).max() or 1
        image = axis.imshow(mean, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{label}: mean score drop when hidden")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(Path(checkpoint).parent.name)
    figure.tight_layout()
    figure.savefig(output_dir / "occlusion.png", dpi=110)
    plt.close(figure)
    (output_dir / "occlusion.json").write_text(json.dumps(stats, indent=2))
    return stats
