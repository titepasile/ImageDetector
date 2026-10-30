"""Stage A, loop 2: train the local encoder on 128x128 native-resolution crops.

Supervised training: LocalResNet + temporary fc head, cross-entropy
on real/AI labels. After training, only the encoder part (model.features) is
used further; export_local_features.py writes the z2 vectors for stage B.

What happens per epoch:
  1. train   forward -> loss -> backward -> update, over all training crops
  2. val     predictions on the validation split, crop- and image-level
  3. save    last.pt every epoch, best.pt when the validation image AUC
             is the best so far
  4. log     console, OUT/metrics.jsonl and (optional) MLflow

--> validation, checkpoints, resuming, a per-step learning-rate schedule (warmup + cosine), mixed
precision on GPU, and per-image loss weights.

Usage:
    python train_local.py SHARDS_DIR runs/local_v1 --epochs 30 --batch-size 128
    python train_local.py SHARDS_DIR runs/local_v1 --resume      # continue
Smoke test on a few batches:
    python train_local.py SHARDS_DIR runs/smoke --epochs 3 --max-steps 20
"""

import math
import torch
import torch.nn.functional as F

from local_metrics import PredictionCollector


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_cross_entropy(logits, labels, weights):
    """Cross-entropy where every crop counts with its image weight."""
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum()


def warmup_cosine(total_steps, warmup_steps):
    """Learning-rate factor per step: linear warmup, then cosine down to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return factor


def train_epoch(model, loader, optimizer, scheduler, scaler, device, amp_dtype, max_steps):
    model.train()
    loss_sum = weight_sum = correct = seen = 0.0
    steps = 0
    for images, labels, weights, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = weighted_cross_entropy(logits.float(), labels, weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item() * weights.sum().item()
        weight_sum += weights.sum().item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()
        steps += 1
        if max_steps and steps >= max_steps:
            break
    if steps == 0:
        raise RuntimeError("Training loader produced no batches (dataset smaller than batch size?)")
    return {"train_loss": loss_sum / weight_sum, "train_crop_accuracy": correct / seen,
            "train_steps": steps, "lr": scheduler.get_last_lr()[0]}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, prefix="val_", max_steps=0):
    """Return metrics dict and the collector with all predictions."""
    model.eval()
    collector = PredictionCollector()
    loss_sum = weight_sum = 0.0
    for step, (images, labels, weights, image_ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        weights = weights.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images).float()
        loss_sum += weighted_cross_entropy(logits, labels, weights).item() * weights.sum().item()
        weight_sum += weights.sum().item()
        collector.update(image_ids, labels.cpu().tolist(), logits.softmax(1)[:, 1].cpu().tolist())
        if max_steps and step + 1 >= max_steps:
            break
    metrics = collector.metrics(prefix=prefix)
    if weight_sum:
        metrics[f"{prefix}loss"] = loss_sum / weight_sum
    return metrics, collector


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, metrics, args):
    torch.save({
        "model_config": model.config, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(), "epoch": epoch, "metrics": metrics,
        "args": vars(args), "feature_dim": model.feature_dim,
    }, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
