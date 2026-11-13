"""Train a detector: python -m scripts.train model=dual_resolution data=synthetic."""

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from src.training import run_training


def absolute_paths(config):
    if config.data.name == "manifest":
        config.data.manifest = to_absolute_path(config.data.manifest)
    for key in (
        "resume",
        "pretrained_reference",
        "checkpoint",
        "input",
        "eval_manifest",
    ):
        if config[key]:
            config[key] = to_absolute_path(config[key])
    return config


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config):
    run_training(absolute_paths(config), HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
