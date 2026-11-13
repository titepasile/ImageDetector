"""Classification metrics and predictions shared by validation and the eval CLI."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from .runtime import move_batch


def classification_metrics(records):
    targets = np.array([r["target"] for r in records])
    scores = np.array([r["score"] for r in records])
    predicted = scores >= 0.5
    real, generated = targets == 0, targets == 1
    metrics = {
        "accuracy": float((predicted == targets).mean()),
        "samples": len(records),
        "auroc": None,
        "average_precision": None,
        "balanced_accuracy": None,
        "fpr": float(predicted[real].mean()) if real.any() else None,
        "tpr": float(predicted[generated].mean()) if generated.any() else None,
    }
    if real.any() and generated.any():
        metrics["auroc"] = roc_auc_score(targets, scores)
        metrics["average_precision"] = average_precision_score(targets, scores)
        metrics["balanced_accuracy"] = (1 - metrics["fpr"] + metrics["tpr"]) / 2
    families = sorted({r["generator_family"] for r in records if r["target"] == 1})
    aucs = []
    for family in families:
        subset = [
            r for r in records if r["target"] == 0 or r["generator_family"] == family
        ]
        if real.any():
            aucs.append(
                roc_auc_score(
                    [r["target"] for r in subset], [r["score"] for r in subset]
                )
            )
    metrics["macro_generator_auroc"] = float(np.mean(aucs)) if aucs else None
    return metrics


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    records = []
    loss_sum = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        logits = output["logits"]
        loss_sum += F.cross_entropy(logits, batch["target"], reduction="sum").item()
        scores = logits.softmax(dim=1)[:, 1].cpu().tolist()
        valid = output.get(
            "local_valid", torch.zeros(len(scores), dtype=torch.bool, device=device)
        )
        for i, score in enumerate(scores):
            records.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "group_id": batch["group_id"][i],
                    "source_domain": batch["source_domain"][i],
                    "generator_family": batch["generator_family"][i],
                    "target": int(batch["target"][i]),
                    "score": score,
                    "route": "fused" if bool(valid[i]) else "global_only",
                }
            )
    if not records:
        raise ValueError("Cannot evaluate an empty loader")
    metrics = classification_metrics(records)
    metrics["loss"] = loss_sum / len(records)
    metrics["local_coverage"] = sum(r["route"] == "fused" for r in records) / len(
        records
    )
    for route in ("global_only", "fused"):
        subset = [r for r in records if r["route"] == route]
        if subset:
            metrics.update(
                {
                    f"{route}/{key}": value
                    for key, value in classification_metrics(subset).items()
                }
            )
    return metrics, records
