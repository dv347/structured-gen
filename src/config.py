from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Type, TypeVar

from paths import EVAL_CONFIGS_DIR, TRAIN_CONFIGS_DIR

DEFAULT_SEEDS = [42, 123, 2025, 999, 7]


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
    use_instruction: bool = True


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
    num_train_epochs: int
    per_device_train_batch_size: int
    bf16: bool
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    logging_steps: int
    logging_first_step: bool
    save_strategy: str
    save_steps: int
    eval_strategy: str
    eval_delay: int
    eval_steps: int


@dataclass
class StageConfig(ABC):
    name: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StageConfig":
        stage_classes = {
            "baseline": BaselineConfig,
            "baseline_bnf": BaselineBnfConfig,
            "induction": InductionConfig,
            "structured_reasoning": StructuredReasoningConfig,
            "unified": UnifiedConfig
        }
        stage = data["name"]
        return stage_classes[stage](**data)
    
    def grammar(self) -> str | ModelConfig | None:
        return getattr(self, "grammar_source", None)

    def embeddings(self) -> bool:
        return getattr(self, "use_embeddings", False)


@dataclass
class BaselineConfig(StageConfig):
    pass


@dataclass
class BaselineBnfConfig(StageConfig):
    grammar_source: str


@dataclass
class InductionConfig(StageConfig):
    grammar_source: str


@dataclass
class StructuredReasoningConfig(StageConfig):
    grammar_source: str | ModelConfig
    use_embeddings: bool

    def __post_init__(self):
        if isinstance(self.grammar_source, dict):
            self.grammar_source = ModelConfig(**self.grammar_source)


@dataclass
class UnifiedConfig(StageConfig):
    grammar_source: dict
    use_embeddings: bool


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
    seed: int = 42
    
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
            training_args=training_args,
            seed=data.get("seed", 42), # Use default if not provided
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
    seed: int = 42

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
            stage_two_training=stage_two_training,
            seed=data.get("seed", 42), # Use default if not provided
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


def load_configs(mode: str, path: str, multi_seed: bool) -> List[LoadableConfig]:
    assert mode in ["train", "eval"], f"Invalid mode: {mode}"

    def process_grammar_source(input: dict, seed: int) -> None:
        if multi_seed:
            source = input["stage"].get("grammar_source", None)
            if type(source) == dict:
                path = source.get("path", None)
                if path:
                    input["stage"]["grammar_source"]["path"] = f"{path}_{seed}"

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
        seeds = DEFAULT_SEEDS if multi_seed else [data.get("seed", 42)]
        if mode == "train":
            stage_type = data["stage"]["name"]
            config_class = TwoStageConfig if stage_type == "unified" else TrainingConfig
            for seed in seeds:
                data_copy = copy.deepcopy(data)
                data_copy["seed"] = seed
                data_copy["output_dir"] = f"{data['output_dir']}_{seed}"
                process_grammar_source(data_copy, seed)
                configs.append(config_class.from_dict(data_copy))
        elif mode == "eval":
            experiment_names = [f"{data['experiment_name']}_{seed}" for seed in DEFAULT_SEEDS] if multi_seed else [data["experiment_name"]]
            model_paths = [f"{data['model']['path']}_{seed}" for seed in DEFAULT_SEEDS] if multi_seed else [data["model"]["path"]]
            for experiment_name, model_path, seed in zip(experiment_names, model_paths, seeds):
                data_copy = copy.deepcopy(data)
                data_copy["experiment_name"] = experiment_name
                data_copy["model"]["path"] = model_path
                process_grammar_source(data_copy, seed)
                configs.append(ExperimentConfig.from_dict(data_copy))
    return configs