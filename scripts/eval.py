"""Evaluate a saved checkpoint; data and transforms default to its saved config."""

import hydra
from hydra.core.hydra_config import HydraConfig

from scripts.train import absolute_paths
from src.prediction import run_prediction


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config):
    run_prediction(absolute_paths(config), HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
