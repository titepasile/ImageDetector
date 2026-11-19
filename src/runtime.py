"""Reproducibility, local checkpoints, and explicit MLflow experiment tracking."""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import zipfile

import mlflow
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed, deterministic=True, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            "This implementation supports CPU and CUDA (spectral FFTs require float32)"
        )
    return device


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Checkpoints contain optimizer and RNG state: only load files you trust.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return state


def config_signature(config):
    """Fields that must agree for an exact epoch-boundary resume."""
    data = OmegaConf.to_container(config, resolve=True)
    return {
        key: data[key]
        for key in (
            "seed",
            "deterministic",
            "threads",
            "device",
            "task",
            "model",
            "spectral",
            "views",
            "data",
            "optimizer",
            "loss",
            "loader",
            "selection",
            "augment",
        )
        if key in data
    }


def source_snapshot(directory):
    paths = []
    for folder in ("src", "scripts", "configs"):
        paths.extend(
            p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml")
        )
    paths.extend(ROOT / name for name in ("requirements.txt", "README.md"))
    digest = hashlib.sha256()
    with zipfile.ZipFile(
        directory / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(paths):
            relative = str(path.relative_to(ROOT))
            contents = path.read_bytes()
            digest.update(relative.encode())
            digest.update(contents)
            archive.writestr(relative, contents)
    return digest.hexdigest()


def flatten_parameters(value, prefix=""):
    parameters = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            parameters.update(flatten_parameters(item, name))
        else:
            parameters[name] = str(item)
    return parameters


@contextmanager
def tracked_run(config, directory, kind):
    directory = Path(directory)
    if (directory / "config.yaml").exists():
        raise FileExistsError(
            f"Run directory already contains an experiment: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    tracking_root = ROOT / "logs" / "mlflow"
    tracking_root.mkdir(parents=True, exist_ok=True)
    uri = config.tracking.uri or f"sqlite:///{tracking_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(config.tracking.experiment)
    if experiment is None:
        # Leave artifact placement to a remote server. Local default is self-contained.
        artifact_location = (
            (tracking_root / "artifacts").as_uri() if not config.tracking.uri else None
        )
        experiment_id = mlflow.create_experiment(
            config.tracking.experiment, artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{kind}-{config.model.name}-s{config.seed}",
    ) as run:
        OmegaConf.save(config, directory / "config.yaml", resolve=True)
        parameters = flatten_parameters(OmegaConf.to_container(config, resolve=True))
        mlflow.log_params(parameters)
        source_hash = source_snapshot(directory)
        environment = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
        )
        (directory / "environment.txt").write_text("\n".join(environment) + "\n")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        )
        mlflow.set_tags(
            {
                "task": kind,
                "data_kind": config.data.name,
                "source_sha256": source_hash,
                "git_revision": revision.stdout.strip(),
                "git_dirty": bool(status.stdout.strip()),
            }
        )
        (directory / "run.json").write_text(
            json.dumps({"run_id": run.info.run_id, "tracking_uri": uri}, indent=2)
        )
        for filename in ("config.yaml", "environment.txt", "source.zip", "run.json"):
            mlflow.log_artifact(str(directory / filename))
        if (directory / ".hydra").exists():
            mlflow.log_artifacts(str(directory / ".hydra"), "hydra")
        yield {"run_id": run.info.run_id, "source_sha256": source_hash}


def log_metrics(metrics, step=None):
    finite = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and np.isfinite(value)
    }
    mlflow.log_metrics(finite, step=step)
