import os

import torch
from config import DatasetPaths, InductionConfig, LoraArgs, ModelConfig, StructuredReasoningConfig, TrainingArgs, TrainingConfig, TwoStageConfig, UnifiedConfig
from logger import logger
from pipelines import TrainingPipeline


class UnifiedPipeline:
    def __init__(
        self,
        stage_config: UnifiedConfig,
        model_path: str,
        output_dir: str,
        stage_one_dataset: DatasetPaths,
        stage_two_dataset: DatasetPaths,
        stage_one_lora: LoraArgs,
        stage_two_lora: LoraArgs,
        stage_one_training: TrainingArgs,
        stage_two_training: TrainingArgs
    ):
        initial_grammar = stage_config.grammar_source["variant"]
        induction_config = InductionConfig("induction", grammar_source=initial_grammar)
        induction_dir = os.path.join(output_dir, "induction_checkpoint")
        self.stage_one_config = TrainingConfig(
            stage_config=induction_config,
            model_path=model_path,
            output_dir=induction_dir,
            dataset_paths=stage_one_dataset,
            lora_args=stage_one_lora,
            training_args=stage_one_training
        )

        model_path = os.path.join(induction_dir, "merged_model")
        if stage_config.grammar_source["use_llm"]:
            struct_config = StructuredReasoningConfig("structured_reasoning", grammar_source=ModelConfig(path=model_path, batch_size=2, assistant_model=None))
        else:
            struct_config = StructuredReasoningConfig("structured_reasoning", grammar_source=initial_grammar)
        self.stage_two_config = TrainingConfig(
            stage_config=struct_config,
            model_path=model_path,
            output_dir=output_dir,
            dataset_paths=stage_two_dataset,
            lora_args=stage_two_lora,
            training_args=stage_two_training
        )

    @staticmethod
    def run_stage(config: TrainingConfig):
        pipeline = TrainingPipeline.from_config(config)
        pipeline.run()


    @classmethod
    def from_config(cls, config: TwoStageConfig) -> "UnifiedPipeline":
        return cls(**vars(config))

    def run(self) -> None:
        logger.info("Running induction stage.")
        UnifiedPipeline.run_stage(self.stage_one_config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.mps.is_available():
            torch.mps.empty_cache()

        logger.info("Running structured reasoning stage.")
        UnifiedPipeline.run_stage(self.stage_two_config)