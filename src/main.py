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

    assert torch.cuda.is_available() or torch.mps.is_available(), "CUDA or MPS must be available."

    configs = load_configs(args.mode, args.config)
    for config in configs:
        if args.mode == "train":
            TrainingPipeline.from_config(config).run()
        elif args.mode == "eval":
            Experiment.from_config(config).run()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()