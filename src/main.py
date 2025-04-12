import argparse

import torch

from config import TwoStageConfig, load_configs
from experiment import Experiment
from paths import set_dataset
from pipelines import TrainingPipeline
from pipelines.unified_pipeline import UnifiedPipeline
from utils import clear_gpu_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="Choose mode: 'train' or 'eval'.")
    parser.add_argument("--config", type = str, required = True, help="Path to the config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'SMCalFLow').")
    parser.add_argument("--multi_seed", action="store_true", help="If set, run 5 seeds instead of one.")
    args = parser.parse_args()

    assert torch.cuda.is_available() or torch.mps.is_available(), "CUDA or MPS must be available."

    set_dataset(args.dataset)

    configs = load_configs(
        mode=args.mode,
        path=args.config,
        multi_seed=args.multi_seed
    )
    for config in configs:
        if args.mode == "train":
            UnifiedPipeline.from_config(config).run() if isinstance(config, TwoStageConfig) else TrainingPipeline.from_config(config).run()
        elif args.mode == "eval":
            Experiment.from_config(config).run()

        clear_gpu_cache()


if __name__ == "__main__":
    main()