"""Metrics for training and evaluation.

Two levels:
  crop level   every 128x128 crop is judged on its own
  image level  the AI-probabilities of all crops of one image are averaged
               first. This is the number that matters: in the end we judge
               images, not crops.
"""

from collections import defaultdict

import numpy as np


def roc_auc(labels, scores):
    """Rank-based AUC (Mann-Whitney U), ties counted as half. None if one class only."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average rank for ties
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(scores)]])
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(labels, scores, prefix=""):
    """Accuracy (threshold 0.5), per-class accuracy and AUC."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) == 0:
        return {}
    predictions = (scores >= 0.5).astype(int)
    result = {
        f"{prefix}accuracy": float((predictions == labels).mean()),
        f"{prefix}n": int(len(labels)),
    }
    for value, name in ((0, "real"), (1, "ai")):
        mask = labels == value
        if mask.any():
            result[f"{prefix}accuracy_{name}"] = float((predictions[mask] == value).mean())
    auc = roc_auc(labels, scores)
    if auc is not None:
        result[f"{prefix}auc"] = auc
    return result


class PredictionCollector:
    """Collect per-crop AI-probabilities, then report crop- and image-level metrics."""

    def __init__(self):
        self.image_ids, self.labels, self.scores = [], [], []

    def update(self, image_ids, labels, ai_probabilities):
        self.image_ids.extend(image_ids)
        self.labels.extend(int(v) for v in labels)
        self.scores.extend(float(v) for v in ai_probabilities)

    def per_image(self):
        """Return {image_id: (label, mean AI-probability, number of crops)}."""
        sums = defaultdict(float)
        counts = defaultdict(int)
        labels = {}
        for image_id, label, score in zip(self.image_ids, self.labels, self.scores):
            sums[image_id] += score
            counts[image_id] += 1
            labels[image_id] = label
        return {i: (labels[i], sums[i] / counts[i], counts[i]) for i in sums}

    def metrics(self, prefix=""):
        result = binary_metrics(self.labels, self.scores, prefix=f"{prefix}crop_")
        images = self.per_image()
        image_labels = [label for label, _, _ in images.values()]
        image_scores = [score for _, score, _ in images.values()]
        result.update(binary_metrics(image_labels, image_scores, prefix=f"{prefix}image_"))
        return result
