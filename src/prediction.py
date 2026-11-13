"""Load self-contained detector checkpoints without downloading pretrained weights."""

import json
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
from PIL import Image
import torch

from .data.dataset import build_datasets, file_digest
from .data.views import ImageViews, canonical_image
from .evaluation import evaluate
from .model.model import build_model
from .runtime import (
    get_device,
    load_checkpoint,
    log_metrics,
    seed_everything,
    tracked_run,
)
from .training import make_loader


def load_detector(path, device):
    state = load_checkpoint(path)
    if state["task"] != "train":
        raise ValueError(
            "Expected a detector checkpoint, not a spectral pretraining checkpoint"
        )
    config = OmegaConf.create(state["config"])
    model = build_model(config, download_weights=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config, state


@torch.inference_mode()
def predict_image(model, path, config, device):
    with Image.open(path) as original:
        image = canonical_image(original)
    sample_id = file_digest(path)
    transform = ImageViews(
        config.views.global_size,
        config.views.local_size,
        config.views.patches,
        config.seed,
    )
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in transform(image, sample_id).items()
    }
    output = model(batch)
    score = output["logits"].softmax(dim=1)[0, 1].item()
    valid = output.get("local_valid", torch.tensor([False]))[0].item()
    return {
        "path": str(path),
        "sample_id": sample_id,
        "score": score,
        "label": "generated" if score >= 0.5 else "real",
        "threshold": 0.5,
        "calibrated": False,
        "route": "fused" if valid else "global_only",
        "valid_patches": (
            int(batch["local_mask"].sum()) if "local_valid" in output else 0
        ),
        "crop_boxes": (
            batch["crop_boxes"][0].cpu().tolist() if "local_valid" in output else []
        ),
    }


def run_prediction(config, directory, inference=False):
    if not config.checkpoint:
        raise ValueError("Set checkpoint=/path/to/best.pt")
    seed_everything(config.seed, config.deterministic, config.threads)
    device = get_device(config.device)
    model, saved_config, state = load_detector(config.checkpoint, device)
    directory = Path(directory)
    # Log the model/data actually used, not the CLI's default training model.
    effective_config = OmegaConf.create(state["config"])
    for key in (
        "tracking",
        "device",
        "threads",
        "checkpoint",
        "input",
        "eval_split",
        "eval_manifest",
    ):
        effective_config[key] = config[key]
    effective_config.task = "inference" if inference else "evaluation"
    if config.eval_manifest and not inference:
        effective_config.data = OmegaConf.create(
            {
                "name": "manifest",
                "manifest": config.eval_manifest,
                "train_split": saved_config.data.train_split,
                "val_split": saved_config.data.val_split,
                "test_split": saved_config.data.test_split,
            }
        )
    with tracked_run(effective_config, directory, effective_config.task):
        mlflow.set_tags(
            {
                "checkpoint_run_id": state["run_id"],
                "checkpoint_sha256": file_digest(config.checkpoint),
                "detector_model": saved_config.model.name,
            }
        )
        mlflow.log_dict(state["config"], "detector_config.json")
        if inference:
            if not config.input:
                raise ValueError("Set input=/path/to/image-or-directory")
            path = Path(config.input)
            paths = (
                sorted(p for p in path.rglob("*") if p.is_file())
                if path.is_dir()
                else [path]
            )
            if not paths:
                raise ValueError("Input directory is empty")
            records = []
            for path in paths:
                try:
                    records.append(predict_image(model, path, saved_config, device))
                except (OSError, ValueError) as exc:
                    records.append({"path": str(path), "error": str(exc)})
        else:
            if config.eval_manifest:
                saved_config.data = OmegaConf.create(
                    {
                        "name": "manifest",
                        "manifest": config.eval_manifest,
                        "train_split": saved_config.data.train_split,
                        "val_split": saved_config.data.val_split,
                        "test_split": saved_config.data.test_split,
                    }
                )
            split = config.eval_split or saved_config.data.test_split
            datasets, fingerprint = build_datasets(saved_config, evaluation_split=split)
            if split not in datasets:
                raise ValueError(f"Evaluation split {split!r} is not available")
            loader = make_loader(datasets[split], saved_config)
            metrics, records = evaluate(model, loader, device)
            log_metrics(metrics)
            mlflow.set_tags(
                {"evaluation_split": split, "evaluation_data_sha256": fingerprint}
            )
            mlflow.log_dict(metrics, "metrics.json")
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
        path = directory / "predictions.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        mlflow.log_artifact(str(path))
        if any("error" in record for record in records):
            raise ValueError(f"Some inputs failed; see {path}")
    return path
