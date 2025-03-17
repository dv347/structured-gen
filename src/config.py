from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Type, TypeVar

from paths import EVAL_CONFIGS_DIR, TRAIN_CONFIGS_DIR


@dataclass
class ModelConfig:
    path: str
    batch_size: int
    assistant_model: str | None = None # Batch inference does not support assistant models


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
class DatasetPaths:
    train_path: str
    val_path: str


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
            "structured_reasoning": StructuredReasoningConfig,
            "unified": UnifiedConfig
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
    grammar_source: str | ModelConfig

    def __post_init__(self):
        if isinstance(self.grammar_source, dict):
            self.grammar_source = ModelConfig(**self.grammar_source)


@dataclass
class UnifiedConfig(StageConfig):
    grammar_source: dict


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
    model_path: str
    output_dir: str
    dataset_paths: DatasetPaths
    lora_args: LoraArgs
    training_args: TrainingArgs
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        stage_config = StageConfig.from_dict(data["stage"])
        dataset_paths = DatasetPaths(**data["dataset"])
        lora_args = LoraArgs(**data["lora_args"])
        training_args = TrainingArgs(**data["training_args"])

        return cls(
            stage_config=stage_config,
            model_path=data["model_path"],
            output_dir=data["output_dir"],
            dataset_paths=dataset_paths,
            lora_args=lora_args,
            training_args=training_args
        )


@dataclass
class TwoStageConfig(LoadableConfig):
    stage_config: UnifiedConfig
    model_path: str
    output_dir: str
    stage_one_dataset: DatasetPaths
    stage_two_dataset: DatasetPaths
    stage_one_lora: LoraArgs
    stage_two_lora: LoraArgs
    stage_one_training: TrainingArgs
    stage_two_training: TrainingArgs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwoStageConfig":
        stage_config = StageConfig.from_dict(data["stage"])
        stage_one_dataset = DatasetPaths(**data["stage_one"]["dataset"])
        stage_two_dataset = DatasetPaths(**data["stage_two"]["dataset"])
        stage_one_lora = LoraArgs(**data["stage_one"]["lora_args"])
        stage_two_lora = LoraArgs(**data["stage_two"]["lora_args"])
        stage_one_training = TrainingArgs(**data["stage_one"]["training_args"])
        stage_two_training = TrainingArgs(**data["stage_two"]["training_args"])

        return cls(
            stage_config=stage_config,
            model_path=data["model_path"],
            output_dir=data["output_dir"],
            stage_one_dataset=stage_one_dataset,
            stage_two_dataset=stage_two_dataset,
            stage_one_lora=stage_one_lora,
            stage_two_lora=stage_two_lora,
            stage_one_training=stage_one_training,
            stage_two_training=stage_two_training
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

    path = os.path.join(TRAIN_CONFIGS_DIR, path) if mode == "train" else os.path.join(EVAL_CONFIGS_DIR, path)

    if os.path.isdir(path):
        config_paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".json")]
    elif os.path.isfile(path):
        config_paths = [path]
    else:
        raise FileNotFoundError(f"Invalid path: {path} does not exist.")

    configs = []
    for config_path in config_paths:
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if mode == "train":
            stage_type = data["stage"]["name"]
            config_class = TwoStageConfig if stage_type == "unified" else TrainingConfig
            configs.append(config_class.from_dict(data))
        elif mode == "eval":
            configs.append(ExperimentConfig.from_dict(data))
    return configs