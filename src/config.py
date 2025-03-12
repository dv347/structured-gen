from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Type, TypeVar

from paths import EVAL_CONFIGS_DIR, TRAIN_CONFIGS_DIR


@dataclass
class ModelConfig:
    path: str
    assistant_model: str


@dataclass
class PromptConfig(ABC):
    strategy: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptConfig":
        classes = {
            "zero-shot": ZeroShotConfig,
            "few-shot": FewShotConfig
        }
        strategy = data["strategy"]
        return classes[strategy](**data)


@dataclass
class ZeroShotConfig(PromptConfig):
    pass


@dataclass
class FewShotConfig(PromptConfig):
    k: int
    exemplars_path: str


@dataclass
class LoraArgs:
    rank_dimension: int # Rank dimension - typically between 4-32
    lora_alpha: int # LoRA scaling factor - typically 2x rank
    lora_dropout: float # Dropout probability for LoRA layers
    bias: str # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules: str # Which modules to apply LoRA to


@dataclass
class TrainingArgs:
    max_steps: int
    per_device_train_batch_size: int
    bf16: bool
    learning_rate: float
    logging_steps: int
    eval_strategy: str
    save_steps: int
    eval_steps: int


@dataclass
class StageConfig(ABC):
    name: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StageConfig":
        stage_classes = {
            "baseline": BaselineConfig,
            "induction": InductionConfig,
            "structured_reasoning": StructuredReasoningConfig
        }
        stage = data["name"]
        return stage_classes[stage](**data)


@dataclass
class BaselineConfig(StageConfig):
    pass


@dataclass
class InductionConfig(StageConfig):
    grammar_source: str


@dataclass
class StructuredReasoningConfig(StageConfig):
    grammar_source: str | dict


T = TypeVar("T", bound="LoadableConfig")


@dataclass
class LoadableConfig(ABC):
    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        """Load a configuration from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)
    
    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a config instance from a dictionary."""
        raise NotImplementedError("Override me!")


@dataclass
class TrainingConfig(LoadableConfig):
    stage_config: StageConfig
    model_name: str
    output_dir: str
    train_path: str
    val_path: str
    lora_args: LoraArgs
    training_args: TrainingArgs
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        stage_config = StageConfig.from_dict(data["stage"])
        lora_args = LoraArgs(**data["lora_args"])
        training_args = TrainingArgs(**data["training_args"])

        return cls(
            stage_config=stage_config,
            model_name=data["model_name"],
            output_dir=data["output_dir"],
            train_path=data["train_path"],
            val_path=data["val_path"],
            lora_args=lora_args,
            training_args=training_args
        )


@dataclass
class ExperimentConfig(LoadableConfig):
    experiment_name: str
    stage_config: StageConfig
    model_config: ModelConfig
    prompt_config: PromptConfig
    test_set_path: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        stage_config = StageConfig.from_dict(data["stage"])
        model_config = ModelConfig(**data["model"])
        prompt_config = PromptConfig.from_dict(data["prompt_strategy"])

        return cls(
            experiment_name=data["experiment_name"],
            stage_config=stage_config,
            model_config=model_config,
            prompt_config=prompt_config,
            test_set_path=data["test_set_path"]
        )
    

def load_configs(mode: str, path: str) -> List[LoadableConfig]:
    assert mode in ["train", "eval"], f"Invalid mode: {mode}"

    mode_classes: Dict[str, LoadableConfig] = {
        "train": TrainingConfig,
        "eval": ExperimentConfig
    }

    path = os.path.join(TRAIN_CONFIGS_DIR, path) if mode == "train" else os.path.join(EVAL_CONFIGS_DIR, path)

    if os.path.isdir(path):
        config_paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".json")]
    elif os.path.isfile(path):
        config_paths = [path]
    else:
        raise ValueError(f"Invalid path: {path} does not exist.")

    configs = []
    for config_path in config_paths:
        configs.append(mode_classes[mode].from_file(config_path))
    return configs