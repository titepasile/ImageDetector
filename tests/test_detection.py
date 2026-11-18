import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from src.data.dataset import build_datasets, group_weights, read_manifest
from src.data.views import ImageViews, image_tensor
from src.evaluation import classification_metrics, evaluate
from src.model.model import build_model
from src.model.spectral import SpectralReference, split_frequencies
from src.prediction import load_detector, predict_image
from src.runtime import load_checkpoint, seed_everything
from src.training import detection_loss, make_loader, run_training

ROOT = Path(__file__).resolve().parents[1]


def configuration(*overrides, pretrain=False):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="pretrain" if pretrain else "config",
            overrides=["experiment=smoke", *overrides],
        )


class ViewAndModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)

    def test_imagenet_checkpoint_construction_does_not_download_weights(self):
        from torchvision.models import resnet18

        cfg = configuration("model=dual_imagenet")
        with patch("torchvision.models.resnet18", wraps=resnet18) as factory:
            model = build_model(cfg, download_weights=False)
        self.assertIsNone(factory.call_args.kwargs["weights"])
        datasets, _ = build_datasets(cfg)
        batch = next(iter(make_loader(datasets[cfg.data.val_split], cfg)))
        model.eval()
        with torch.inference_mode():
            self.assertEqual(model(batch)["logits"].shape, (4, 2))

    def test_native_crops_and_small_image_masks(self):
        pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        image = Image.fromarray(pixels)
        views = ImageViews(32, 32, 4, 17)
        result = views(image, "sample")
        for patch, box in zip(result["local"], result["crop_boxes"]):
            x, y, width, height = box.tolist()
            torch.testing.assert_close(
                patch,
                image_tensor(image.crop((x, y, x + width, y + height))),
                rtol=0,
                atol=0,
            )
        self.assertEqual(
            int(views(Image.new("RGB", (32, 32)), "exact")["local_mask"].sum()), 1
        )
        self.assertFalse(
            views(Image.new("RGB", (16, 64)), "narrow")["local_mask"].any()
        )

    def test_crops_repeat_across_worker_counts_but_change_with_epoch(self):
        cfg = configuration("views.patches=4")
        datasets, _ = build_datasets(cfg)
        dataset = datasets[cfg.data.train_split]
        first = {
            b["sample_id"][i]: b["crop_boxes"][i]
            for b in make_loader(dataset, cfg, 0, True)
            for i in range(len(b["sample_id"]))
        }
        cfg.loader.workers = 2
        second = {
            b["sample_id"][i]: b["crop_boxes"][i]
            for b in make_loader(dataset, cfg, 0, True)
            for i in range(len(b["sample_id"]))
        }
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        dataset.set_epoch(1)
        self.assertTrue(
            any(
                not torch.equal(first[item["sample_id"]], item["crop_boxes"])
                for item in dataset
            )
        )

    def test_dual_gradients_fallback_and_frozen_batchnorm(self):
        cfg = configuration("model=dual_resolution")
        datasets, _ = build_datasets(cfg)
        batch = next(iter(make_loader(datasets[cfg.data.train_split], cfg)))
        model = build_model(cfg)
        output = model(batch)
        detection_loss(output, batch, cfg.loss).backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.local_encoder.parameters()
            )
        )
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.global_encoder.parameters()
            )
        )
        batch["local_mask"].zero_()
        model.eval()
        output = model(batch)
        torch.testing.assert_close(output["logits"], output["global_logits"])
        self.assertTrue(torch.isfinite(output["logits"]).all())

        cfg.model.freeze_global = True
        frozen = build_model(cfg)
        before = {k: v.clone() for k, v in frozen.global_encoder.state_dict().items()}
        frozen.train()
        detection_loss(frozen(batch), batch, cfg.loss).backward()
        for key, value in frozen.global_encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_spectral_objectives_and_deterministic_probe_features(self):
        patches = torch.rand(3, 3, 32, 32)
        low, high = split_frequencies(patches, 0.125)
        torch.testing.assert_close(low + high, patches)
        torch.testing.assert_close(
            split_frequencies(high, 0.125)[0], torch.zeros_like(high), atol=1e-6, rtol=0
        )
        for objective in ("reconstruction", "flow"):
            model = SpectralReference(objective, channels=8, n_blocks=1)
            loss = model.training_loss(patches)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters()
                )
            )
            model.eval()
            whole = model.evidence(patches)
            separate = torch.cat([model.evidence(patch[None]) for patch in patches])
            torch.testing.assert_close(whole, separate, atol=1e-5, rtol=1e-5)
        sampled = model.sample(patches, steps=2)
        torch.testing.assert_close(
            split_frequencies(sampled, 0.125)[0], low, atol=1e-6, rtol=1e-5
        )

    def test_real_only_pretraining_preserves_sample_identity(self):
        cfg = configuration()
        all_data, fingerprint = build_datasets(cfg)
        real_data, real_fingerprint = build_datasets(cfg, real_only=True)
        self.assertEqual(fingerprint, real_fingerprint)
        original = {item["sample_id"]: item for item in all_data[cfg.data.train_split]}
        for item in real_data[cfg.data.train_split]:
            self.assertEqual(item["target"], 0)
            torch.testing.assert_close(
                item["global"], original[item["sample_id"]]["global"]
            )

    def test_group_weights_and_known_metrics(self):
        records = [
            dict(label=0, group_id="r", source_domain="camera"),
            dict(label=1, group_id="a", generator_family="a"),
            dict(label=1, group_id="b", generator_family="b"),
            dict(label=1, group_id="b", generator_family="b"),
        ]
        weights = group_weights(records)
        self.assertAlmostEqual(weights[0], sum(weights[1:]))
        self.assertAlmostEqual(weights[1], sum(weights[2:]))
        predictions = [
            dict(target=y, score=s, generator_family="g" if y else "real")
            for y, s in [(0, 0.1), (0, 0.4), (1, 0.6), (1, 0.9)]
        ]
        metrics = classification_metrics(predictions)
        self.assertEqual(metrics["auroc"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 1)
        self.assertIsNone(classification_metrics(predictions[:2])["auroc"])


def release_mlflow_database(uri):
    """Close MLflow's cached SQLite engine so Windows can delete the database file."""
    mlflow.end_run()
    try:
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    except ImportError:
        return
    # MLflow keeps one engine per URI for the whole process (private API).
    engine = getattr(SqlAlchemyStore, "_engine_map", {}).pop(uri, None)
    if engine is not None:
        engine.dispose()


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        # Cleanup errors are still ignored as a fallback for other open handles.
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        # Cleanups run in reverse order: release the database before deleting it.
        self.addCleanup(release_mlflow_database, self.uri)

    def config(self, *overrides, pretrain=False):
        cfg = configuration(*overrides, pretrain=pretrain)
        cfg.tracking.uri = self.uri
        cfg.tracking.experiment = "integration-tests"
        cfg.selection.patience = 0
        return cfg

    def test_epoch_resume_matches_uninterrupted_training_and_tracks_artifacts(self):
        cfg = self.config()
        run_training(cfg, self.root / "full")
        cfg.epochs = 1
        run_training(cfg, self.root / "first")
        cfg.epochs = 2
        cfg.resume = str(self.root / "first" / "last.pt")
        run_training(cfg, self.root / "resumed")
        expected = load_checkpoint(self.root / "full" / "last.pt")
        actual = load_checkpoint(self.root / "resumed" / "last.pt")
        for key, value in expected["model"].items():
            torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
        self.assertEqual(expected["best_value"], actual["best_value"])
        client = MlflowClient(tracking_uri=self.uri)
        run = client.get_run(actual["run_id"])
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.params["seed"], "17")
        self.assertIn("val/auroc", run.data.metrics)
        self.assertIn(
            "source.zip", [a.path for a in client.list_artifacts(run.info.run_id)]
        )
        self.assertIn(
            "training/best.pt",
            [a.path for a in client.list_artifacts(run.info.run_id, "training")],
        )
        cfg.seed += 1
        with self.assertRaisesRegex(ValueError, "configuration or dataset"):
            run_training(cfg, self.root / "invalid-resume")

    def test_both_spectral_references_train_and_transfer_to_detector(self):
        for objective in ("reconstruction", "flow"):
            cfg = self.config(f"spectral={objective}", "epochs=1", pretrain=True)
            reference_path = run_training(cfg, self.root / f"reference-{objective}")
            cfg = self.config(
                "model=dual_resolution", f"spectral={objective}", "epochs=1"
            )
            cfg.pretrained_reference = str(reference_path)
            checkpoint_path = run_training(cfg, self.root / f"detector-{objective}")
            model, saved, _ = load_detector(checkpoint_path, torch.device("cpu"))
            reference = load_checkpoint(reference_path)["model"]
            for key, value in model.reference.state_dict().items():
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in model.reference.parameters()))
            datasets, _ = build_datasets(saved, evaluation_split=saved.data.test_split)
            metrics, _ = evaluate(
                model,
                make_loader(datasets[saved.data.test_split], saved),
                torch.device("cpu"),
            )
            self.assertTrue(np.isfinite(metrics["loss"]))

    def test_manifest_leakage_and_evaluation_inference_agreement(self):
        records = []
        for index in range(12):
            path = self.root / f"{index}.png"
            pixels = np.random.default_rng(index).integers(
                0, 255, (40, 48, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(path)
            records.append(
                dict(
                    path=path.name,
                    label=index % 2,
                    split=["train", "val_id", "test_id"][index // 4],
                    group_id=str(index),
                    source_domain="camera",
                    generator_family="toy",
                )
            )
        manifest = self.root / "samples.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        cfg = self.config("epochs=1")
        cfg.data = OmegaConf.create(
            dict(
                name="manifest",
                manifest=str(manifest),
                train_split="train",
                val_split="val_id",
                test_split="test_id",
            )
        )
        path = run_training(cfg, self.root / "manifest-run")
        model, saved, _ = load_detector(path, torch.device("cpu"))
        datasets, _ = build_datasets(saved, evaluation_split="test_id")
        _, predictions = evaluate(
            model, make_loader(datasets["test_id"], saved), torch.device("cpu")
        )
        prediction = predict_image(
            model, self.root / "8.png", saved, torch.device("cpu")
        )
        self.assertAlmostEqual(prediction["score"], predictions[0]["score"], places=6)
        records[4]["group_id"] = records[0]["group_id"]
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
