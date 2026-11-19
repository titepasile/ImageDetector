import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from scripts.experiments import merge, subset
from src.data.dataset import augment_image
from src.data.perturb import apply_operation, perturb_manifest
from src.suite import bootstrap, calibrated_summary, threshold_at_fpr, threshold_balanced
from src.training import detection_loss, learning_rate
from tests.test_detection import configuration
from src.model.model import build_model


def noise_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


class ModelAndTrainingTests(unittest.TestCase):
    def test_local_only_model_gradients_and_fallback(self):
        cfg = configuration("model=local_only")
        model = build_model(cfg)
        batch = {
            "local": torch.rand(3, 1, 3, 32, 32),
            "local_mask": torch.tensor([[True], [True], [False]]),
            "target": torch.tensor([0, 1, 1]),
            "loss_weight": torch.ones(3),
        }
        output = model(batch)
        self.assertEqual(output["logits"].shape, (3, 2))
        torch.testing.assert_close(output["logits"][2], torch.zeros(2))
        detection_loss(output, batch, cfg.loss).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_cosine_schedule(self):
        cfg = configuration("optimizer.schedule=cosine", "epochs=4")
        values = [learning_rate(cfg, epoch) for epoch in range(4)]
        self.assertAlmostEqual(values[0], cfg.optimizer.lr)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertEqual(learning_rate(configuration(), 3), configuration().optimizer.lr)

    def test_augmentation_is_deterministic(self):
        settings = OmegaConf.create(
            {"probability": 1.0, "blur_sigma": [1.0, 2.0], "jpeg_quality": [30, 40]}
        )
        image = noise_image()
        first = np.asarray(augment_image(image, settings, "17:0:a"))
        second = np.asarray(augment_image(image, settings, "17:0:a"))
        other = np.asarray(augment_image(image, settings, "17:1:a"))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertFalse(np.array_equal(first, np.asarray(image)))


class PerturbationTests(unittest.TestCase):
    def test_operations_keep_size_and_change_pixels(self):
        image = noise_image()
        for operation in (
            "jpeg:50", "resample:0.75", "blur:1", "lowpass:0.125", "highpass:0.125",
            "shuffle:16", "border_gray:8", "border_reflect:8",
        ):
            result = apply_operation(image, operation, "x")
            self.assertEqual(result.size, image.size, operation)
            self.assertFalse(np.array_equal(np.asarray(result), np.asarray(image)), operation)
        self.assertEqual(apply_operation(image, "crop:32", "x").size, (32, 32))
        shuffled = np.asarray(apply_operation(image, "shuffle:16", "x"))
        self.assertEqual(sorted(shuffled.ravel()), sorted(np.asarray(image).ravel()))
        border = np.asarray(apply_operation(image, "border_gray:8", "x"))
        np.testing.assert_array_equal(border[8:-8, 8:-8], np.asarray(image)[8:-8, 8:-8])

    def test_perturb_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = []
            for index, label in enumerate((0, 1)):
                noise_image(seed=index).save(root / f"{index}.png")
                record = {"path": f"{index}.png", "label": label, "split": "test_id",
                          "group_id": f"g{index}", "source_domain": "d"}
                if label:
                    record["generator_family"] = "toy"
                lines.append(json.dumps(record))
            (root / "samples.jsonl").write_text("\n".join(lines) + "\n")
            count = perturb_manifest(root / "samples.jsonl", "test_id", root / "out", ["jpeg:75"])
            self.assertEqual(count, 2)
            written = [json.loads(l) for l in (root / "out" / "samples.jsonl").read_text().splitlines()]
            self.assertEqual([r["perturbation"] for r in written], [["jpeg:75"]] * 2)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.records = [
            {"target": t, "score": float(s), "generator_family": "g" if t else "real"}
            for t in (0, 1) for s in rng.normal(0.3 + 0.4 * t, 0.1, 200)
        ]

    def test_thresholds(self):
        threshold = threshold_at_fpr(self.records, 0.05)
        real = np.array([r["score"] for r in self.records if r["target"] == 0])
        self.assertLessEqual((real >= threshold).mean(), 0.05)
        self.assertGreater((real >= threshold).mean(), 0.03)
        self.assertTrue(0.35 < threshold_balanced(self.records) < 0.65)

    def test_bootstrap_and_summary(self):
        low, high = bootstrap(self.records, lambda r: np.mean([x["score"] for x in r]), 200)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        summary = calibrated_summary(self.records, self.records, samples=50)
        self.assertLessEqual(summary["auroc_ci"][0], summary["auroc"])
        self.assertLessEqual(summary["val_fpr5_threshold"]["fpr"], 0.05)


class ManifestToolTests(unittest.TestCase):
    def test_subset_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root / name).mkdir()
                lines = []
                for i in range(6):
                    split = "train" if i < 4 else "val_id"
                    for label in (0, 1):
                        record = {"path": f"{i}_{label}.png", "label": label, "split": split,
                                  "group_id": f"{name}{i}{label}", "source_domain": "d",
                                  # Real images 0 and 4 exist in both sources.
                                  "source_member": f"x/{'shared' if i in (0, 4) else name}{i}_{label}.JPEG"}
                        lines.append(json.dumps(record))
                (root / name / "samples.jsonl").write_text("\n".join(lines) + "\n")
            counts = subset(root / "a" / "samples.jsonl", 2, root / "a" / "small.jsonl")
            self.assertEqual(counts, {0: 2, 1: 2})
            small = (root / "a" / "small.jsonl").read_text().splitlines()
            self.assertEqual(len(small), 8)  # 2+2 train, 2+2 validation
            counts = merge(root / "joint.jsonl", [f"{root / n / 'samples.jsonl'}:4:2" for n in ("a", "b")])
            self.assertEqual(counts["skipped_duplicate_real"], 2)
            merged = [json.loads(l) for l in (root / "joint.jsonl").read_text().splitlines()]
            self.assertTrue(all(Path(r["path"]).is_absolute() for r in merged))
            real = [Path(r["source_member"]).name for r in merged if r["label"] == 0]
            self.assertEqual(len(real), len(set(real)))


if __name__ == "__main__":
    unittest.main()
