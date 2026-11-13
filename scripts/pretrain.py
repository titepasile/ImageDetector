"""Train a real-only spectral reference with reconstruction or flow matching."""

import hydra
from hydra.core.hydra_config import HydraConfig

from scripts.train import absolute_paths
from src.training import run_training


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(config):
    run_training(absolute_paths(config), HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
