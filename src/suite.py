"""Evaluate many checkpoints on many test sets, then calibrate and summarise.

Unlike scripts.eval this loads each model once and writes plain files (no MLflow
run per test set). Every result records the checkpoint and manifest hashes.

Layout: <root>/<run>/<test_set>/{predictions.jsonl, metrics.json}
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
import torch

from .data.dataset import build_datasets, file_digest
from .evaluation import classification_metrics, evaluate, tpr_at_fpr
from .prediction import load_detector
from .training import make_loader


def evaluate_checkpoint(checkpoint, test_sets, root, run_name, device="cpu", batch_size=64, workers=4):
    """test_sets: {name: (manifest or None, split)}; None means the run's own manifest."""
    model, config, state = load_detector(checkpoint, device)
    checkpoint_sha = file_digest(checkpoint)
    results = {}
    for name, (manifest, split) in test_sets.items():
        target = Path(root) / run_name / name
        if (target / "metrics.json").exists():
            results[name] = json.loads((target / "metrics.json").read_text())
            continue
        cfg = copy.deepcopy(config)
        if manifest is not None:
            cfg.data = OmegaConf.create(
                {
                    "name": "manifest",
                    "manifest": str(Path(manifest).resolve()),
                    "train_split": config.data.train_split,
                    "val_split": config.data.val_split,
                    "test_split": config.data.test_split,
                }
            )
        cfg.loader.batch_size = batch_size
        cfg.loader.workers = workers
        datasets, fingerprint = build_datasets(cfg, evaluation_split=split)
        loader = make_loader(datasets[split], cfg)
        with torch.inference_mode():
            metrics, records = evaluate(model, loader, device)
        metrics.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            manifest=cfg.data.manifest,
            split=split,
            data_fingerprint=fingerprint,
            training_run_id=state["run_id"],
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results[name] = metrics
    return results


def load_predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def threshold_at_fpr(records, max_fpr=0.05):
    """Smallest threshold whose false-positive rate on these records is <= max_fpr."""
    real = np.sort([r["score"] for r in records if r["target"] == 0])
    index = int(np.ceil((1 - max_fpr) * len(real))) - 1
    candidate = real[min(max(index, 0), len(real) - 1)]
    # Scores equal to the threshold count as positive, so step just above it.
    return float(np.nextafter(candidate, np.inf))


def threshold_balanced(records):
    """Threshold maximising balanced accuracy on these records."""
    scores = np.array([r["score"] for r in records])
    targets = np.array([r["target"] for r in records])
    best = (-1.0, 0.5)
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        value = (predicted[targets == 1].mean() + (~predicted[targets == 0]).mean()) / 2
        if value > best[0]:
            best = (float(value), float(threshold))
    return best[1]


def rates(records, threshold):
    scores = np.array([r["score"] for r in records])
    targets = np.array([r["target"] for r in records])
    predicted = scores >= threshold
    fpr = float(predicted[targets == 0].mean())
    tpr = float(predicted[targets == 1].mean())
    return {"fpr": fpr, "tpr": tpr, "balanced_accuracy": (1 - fpr + tpr) / 2}


def bootstrap(records, statistic, samples=1000, seed=0):
    """Percentile 95% interval; resampling within each class keeps the class balance."""
    rng = np.random.default_rng(seed)
    groups = [[r for r in records if r["target"] == label] for label in (0, 1)]
    values = []
    for _ in range(samples):
        resampled = [g[i] for g in groups for i in rng.integers(0, len(g), len(g))]
        values.append(statistic(resampled))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _auroc(records):
    return float(roc_auc_score([r["target"] for r in records], [r["score"] for r in records]))


def _tpr5(records):
    return tpr_at_fpr([r["target"] for r in records], [r["score"] for r in records])


def calibrated_summary(test_records, val_records, samples=1000):
    """Threshold-free metrics with CIs, plus rates at thresholds chosen on validation."""
    metrics = classification_metrics(test_records)
    summary = {
        "samples": len(test_records),
        "auroc": metrics["auroc"],
        "auroc_ci": bootstrap(test_records, _auroc, samples),
        "average_precision": metrics["average_precision"],
        "tpr_at_fpr_5": metrics["tpr_at_fpr_5"],
        "tpr_at_fpr_5_ci": bootstrap(test_records, _tpr5, samples),
        "default_threshold": rates(test_records, 0.5),
    }
    if val_records is not None:
        for name, threshold in (
            ("val_fpr5_threshold", threshold_at_fpr(val_records)),
            ("val_balanced_threshold", threshold_balanced(val_records)),
        ):
            summary[name] = {"threshold": threshold, **rates(test_records, threshold)}
    return summary


def summarise_root(root, val_name="val", samples=1000):
    """Summaries for every <run>/<test_set>; writes summary.json at the root."""
    root = Path(root)
    summaries = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        val_path = run / val_name / "predictions.jsonl"
        val_records = load_predictions(val_path) if val_path.exists() else None
        for test in sorted(p for p in run.iterdir() if p.is_dir() and p.name != val_name):
            if not (test / "predictions.jsonl").exists():
                continue  # Diagnostics folders (occlusion, spectra) are not test sets.
            records = load_predictions(test / "predictions.jsonl")
            if len({r["target"] for r in records}) < 2:
                continue
            key = f"{run.name}/{test.name}"
            digest = hashlib.sha256((test / "predictions.jsonl").read_bytes()).hexdigest()
            cache = test / "summary.json"
            if cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("predictions_sha256") == digest and (
                    val_records is None or "val_fpr5_threshold" in cached
                ):
                    summaries[key] = cached
                    continue
            summary = calibrated_summary(records, val_records, samples)
            summary["predictions_sha256"] = digest
            cache.write_text(json.dumps(summary, indent=2))
            summaries[key] = summary
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries
