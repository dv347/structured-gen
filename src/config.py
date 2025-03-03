from abc import ABC
from dataclasses import dataclass
import json
import os
from typing import Dict, List, Type

from paths import CONFIGS_DIR


@dataclass
class ModelConfig:
    name: str
    assistant_model: str


@dataclass
class PromptConfig(ABC):
    mode: str


@dataclass
class ZeroShotConfig(PromptConfig):
    pass


@dataclass
class FewShotConfig(PromptConfig):
    k: int
    exemplars_path: str


@dataclass
class ExperimentConfig:
    experiment_name: str
    model_config: ModelConfig
    prompt_config: PromptConfig
    test_set_path: str

    @staticmethod
    def from_file(file_path: str) -> "ExperimentConfig":
        """Load an experiment configuration from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return ExperimentConfig.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, any]) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a dictionary."""
        model_config = ModelConfig(**data["model"])

        prompt_classes = {
            "zero-shot": ZeroShotConfig,
            "few-shot": FewShotConfig
        }
        prompt_class = data["prompt_strategy"]["mode"]
        prompt_config = prompt_classes[prompt_class](**data["prompt_strategy"])

        return ExperimentConfig(
            experiment_name=data["experiment_name"],
            model_config=model_config,
            prompt_config=prompt_config,
            test_set_path=data["test_set_path"]
        )
    

def load_configs(path: str) -> List[ExperimentConfig]:
    if os.path.isdir(path):
        config_paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".json")]
    elif os.path.isfile(path):
        config_paths = [path]
    else:
        raise ValueError(f"Invalid path: {path} does not exist.")

    configs = []
    for config_path in config_paths:
        configs.append(ExperimentConfig.from_file(config_path))
    return configs