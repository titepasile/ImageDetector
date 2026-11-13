"""Predict images with the exact model and transforms stored in a checkpoint."""

import hydra
from hydra.core.hydra_config import HydraConfig

from scripts.train import absolute_paths
from src.prediction import run_prediction


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config):
    run_prediction(
        absolute_paths(config), HydraConfig.get().runtime.output_dir, inference=True
    )


if __name__ == "__main__":
    main()
