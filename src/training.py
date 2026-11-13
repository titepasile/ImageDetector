"""Explicit training loops for classification and real-only spectral pretraining."""

import json
import os
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data.dataset import build_datasets
from .evaluation import evaluate
from .model.model import build_model
from .model.spectral import build_reference
from .online_metrics import OnlineMetrics
from .runtime import (
    config_signature,
    get_device,
    load_checkpoint,
    log_metrics,
    move_batch,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    seed_worker,
    tracked_run,
)


def make_loader(dataset, config, epoch=0, training=False):
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=training,
        num_workers=config.loader.workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
        drop_last=False,
    )


def detection_loss(output, batch, config):
    target = batch["target"]
    per_image = F.cross_entropy(output["logits"], target, reduction="none")
    if "local_valid" in output:
        valid = output["local_valid"].float()
        per_image += (
            valid
            * config.global_aux
            * F.cross_entropy(output["global_logits"], target, reduction="none")
        )
        per_image += (
            valid
            * config.local_aux
            * F.cross_entropy(output["local_logits"], target, reduction="none")
        )
    return (per_image * batch["loss_weight"].float()).mean()


def train_detector_epoch(model, loader, optimizer, config, device):
    model.train()
    metrics = OnlineMetrics(device)
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = detection_loss(output, batch, config.loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite classification loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_grad)
        optimizer.step()
        metrics.update(loss, output["logits"], batch["target"])
    result = metrics.get_dict()
    if result is None:
        raise ValueError("Training loader is empty")
    return {
        "loss": result["train_mean_loss"],
        "accuracy": result["train_accuracy"],
        "batch_loss_variance": result["train_variance"],
    }


def spectral_epoch(model, loader, optimizer, config, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_patches = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            if (batch["target"] != 0).any():
                raise ValueError(
                    "The spectral reference must be trained on real images only"
                )
            patches = batch["local"][batch["local_mask"]]
            if not len(patches):
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(patches)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite spectral loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_grad)
                optimizer.step()
            else:
                # Fixed probes give reproducible selection without consuming training RNG.
                loss = model.evidence(patches).mean()
            total_loss += loss.item() * len(patches)
            total_patches += len(patches)
    if not total_patches:
        raise ValueError("No real images in this split have a valid native patch")
    return {
        "loss" if training else "probe_error": total_loss / total_patches,
        "patches": total_patches,
    }


def load_reference(model, config, fingerprint):
    if model.reference is None:
        return
    if not config.pretrained_reference:
        raise ValueError(
            "This detector needs pretrained_reference=/path/to/spectral/best.pt. "
            "First run scripts.pretrain with the same spectral objective."
        )
    state = load_checkpoint(config.pretrained_reference)
    if state["task"] != "pretrain" or state["data_fingerprint"] != fingerprint:
        raise ValueError(
            "Reference must come from real-only pretraining on the same split inventory"
        )
    expected = OmegaConf.to_container(config.spectral, resolve=True)
    if (
        state["config"]["spectral"] != expected
        or state["config"]["views"]["local_size"] != config.views.local_size
    ):
        raise ValueError(
            "Reference objective, spectral settings, or patch size do not match"
        )
    model.reference.load_state_dict(state["model"])
    mlflow.set_tag("reference_run_id", state["run_id"])


def run_training(config, directory):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed training is not supported yet")
    if config.task not in ("train", "pretrain") or config.epochs < 1:
        raise ValueError("Use task=train/pretrain and epochs >= 1")
    if (
        config.task == "train"
        and config.model.name == "baseline"
        and config.spectral.objective != "none"
    ):
        raise ValueError(
            "Spectral evidence requires model=dual_resolution or model=dual_imagenet"
        )
    if config.selection.mode not in ("min", "max"):
        raise ValueError("selection.mode must be min or max")
    if config.task == "pretrain" and config.spectral.objective == "none":
        raise ValueError(
            "Pretraining requires spectral=reconstruction or spectral=flow"
        )
    seed_everything(config.seed, config.deterministic, config.threads)
    device = get_device(config.device)
    directory = Path(directory)
    with tracked_run(config, directory, config.task) as run:
        pretraining = config.task == "pretrain"
        datasets, fingerprint = build_datasets(config, real_only=pretraining)
        mlflow.set_tag("data_fingerprint", fingerprint)
        inventory = {split: dataset.records for split, dataset in datasets.items()}
        (directory / "inventory.json").write_text(json.dumps(inventory, indent=2))
        mlflow.log_artifact(str(directory / "inventory.json"))
        state = load_checkpoint(config.resume) if config.resume else None
        model = (
            build_reference(config.spectral)
            if pretraining
            else build_model(config, download_weights=state is None)
        )
        if not pretraining and hasattr(model, "reference") and state is None:
            load_reference(model, config, fingerprint)
        model.to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        start_epoch, best_value, bad_epochs = 0, None, 0
        if state is not None:
            if (
                state["signature"] != config_signature(config)
                or state["data_fingerprint"] != fingerprint
            ):
                raise ValueError(
                    "Resume configuration or dataset differs from the saved run"
                )
            if state["source_sha256"] != run["source_sha256"]:
                raise ValueError(
                    "Source code changed; start a new run instead of exact resume"
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            best_value, bad_epochs = state["best_value"], state["bad_epochs"]
            restore_rng(state["rng"])
            mlflow.set_tag("resumed_from", state["run_id"])
            # Preserve the selected model even if this continuation never improves.
            best_path = Path(config.resume).parent / "best.pt"
            if not best_path.is_file():
                raise ValueError(
                    "Resume requires best.pt beside the supplied checkpoint"
                )
            best_state = load_checkpoint(best_path)
            if (
                best_state["signature"] != state["signature"]
                or best_state["data_fingerprint"] != fingerprint
            ):
                raise ValueError("best.pt does not belong to the resumed experiment")
            save_checkpoint(directory / "best.pt", best_state)
        if start_epoch >= config.epochs:
            raise ValueError(
                "epochs is the total target; it must exceed the checkpoint epoch"
            )
        mlflow.log_params(
            {
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )
        mlflow.set_tag("device", str(device))
        history = directory / "metrics.jsonl"
        for epoch in range(start_epoch, config.epochs):
            train_loader = make_loader(
                datasets[config.data.train_split], config, epoch, True
            )
            val_loader = make_loader(datasets[config.data.val_split], config)
            if pretraining:
                training_metrics = spectral_epoch(
                    model, train_loader, optimizer, config, device
                )
                validation_metrics = spectral_epoch(
                    model, val_loader, None, config, device
                )
                value, maximize = validation_metrics["probe_error"], False
            else:
                training_metrics = train_detector_epoch(
                    model, train_loader, optimizer, config, device
                )
                validation_metrics, _ = evaluate(model, val_loader, device)
                value = validation_metrics.get(config.selection.metric)
                maximize = config.selection.mode == "max"
                if value is None:
                    raise ValueError(
                        f"Selection metric {config.selection.metric} is undefined on validation"
                    )
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError("Validation selection metric is not finite")
            metrics = {f"train/{k}": v for k, v in training_metrics.items()}
            metrics.update({f"val/{k}": v for k, v in validation_metrics.items()})
            log_metrics(metrics, step=epoch)
            with history.open("a") as stream:
                stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
            improved = best_value is None or (
                value > best_value if maximize else value < best_value
            )
            best_value = value if improved else best_value
            bad_epochs = 0 if improved else bad_epochs + 1
            checkpoint = {
                "schema_version": 1,
                "task": config.task,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "best_value": best_value,
                "bad_epochs": bad_epochs,
                "config": OmegaConf.to_container(config, resolve=True),
                "signature": config_signature(config),
                "data_fingerprint": fingerprint,
                "label_map": {0: "real", 1: "generated"},
                **run,
            }
            save_checkpoint(directory / "last.pt", checkpoint)
            if improved:
                save_checkpoint(directory / "best.pt", checkpoint)
            print(f"Epoch {epoch + 1}/{config.epochs}: validation={value:.6f}")
            if config.selection.patience and bad_epochs >= config.selection.patience:
                break
        for filename in ("best.pt", "last.pt", "metrics.jsonl"):
            mlflow.log_artifact(str(directory / filename), "training")
    return directory / "best.pt"


def fit(
    model,
    training_split,
    val_split,
    criterion,
    optimizer,
    scheduler,
    scheduler_interv,
    n_epochs,
    device,
):
    """Compatibility loop for tensor-pair loaders inside a caller-owned MLflow run.

    New experiments should use run_training so config, data, source, and random
    state are captured along with metrics and checkpoints.
    """
    if mlflow.active_run() is None:
        raise ValueError(
            "fit requires an active MLflow run; use scripts.train for tracked experiments"
        )
    for i in range(n_epochs):
        metrics = train_epoch(model, training_split, criterion, optimizer, device)
        if metrics is None:
            raise ValueError("Training loader is empty")
        log_metrics(metrics, step=i)
        model.eval()
        validation = OnlineMetrics(device)
        with torch.inference_mode():
            for x, y in val_split:
                x, y = x.to(device), y.to(device)
                out = model(x)
                validation.update(criterion(out, y), out, y)
        val_metrics = validation.get_dict()
        if val_metrics is None:
            raise ValueError("Validation loader is empty")
        log_metrics(
            {
                key.replace("train_", "val_", 1): value
                for key, value in val_metrics.items()
            },
            step=i,
        )
        if scheduler is not None:
            if scheduler_interv != "epoch":
                raise ValueError(
                    "Legacy fit supports epoch schedulers; use run_training for experiments"
                )
            scheduler.step()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    metrics = OnlineMetrics(device)

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        metrics.update(loss, out, y)

    return metrics.get_dict()
