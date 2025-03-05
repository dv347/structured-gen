import argparse

import torch

from config import load_configs
from experiment import Experiment
from training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="Choose mode: 'train' or 'eval'.")
    parser.add_argument("--config", type = str, required = True, help="Path to the config file.")
    args = parser.parse_args()

    configs = load_configs(args.mode, args.config)
    for config in configs:
        if args.mode == "train":
            TrainingPipeline.from_config(config).run()
        elif args.mode == "eval":
            Experiment.from_config(config).run()

        torch.mps.empty_cache()


if __name__ == "__main__":
    main()