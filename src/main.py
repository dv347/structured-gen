import argparse
import os

import torch

from config import load_configs
from experiment import Experiment
from paths import CONFIGS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    args = parser.parse_args()
    path = os.path.join(CONFIGS_DIR, args.config)

    configs = load_configs(path)
    for config in configs:
        Experiment.from_config(config).run()
        torch.mps.empty_cache()