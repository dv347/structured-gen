import os
from config import DatasetPaths, InductionConfig, LoraArgs, StructuredReasoningConfig, TrainingArgs, TrainingConfig, TwoStageConfig, UnifiedConfig
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
        stage_one_config = TrainingConfig(
            stage_config=induction_config,
            model_path=model_path,
            output_dir=induction_dir,
            dataset_paths=stage_one_dataset,
            lora_args=stage_one_lora,
            training_args=stage_one_training
        )

        model_path = os.path.join(induction_dir, "merged_model")
        if stage_config.grammar_source["use_llm"]:
            struct_config = StructuredReasoningConfig("structured_reasoning", grammar_source={"llm": model_path})
        else:
            struct_config = InductionConfig(grammar_source=initial_grammar)
        stage_two_config = TrainingConfig(
            stage_config=struct_config,
            model_path=model_path,
            output_dir=output_dir,
            dataset_paths=stage_two_dataset,
            lora_args=stage_two_lora,
            training_args=stage_two_training
        )
        self.stage_one = TrainingPipeline.from_config(stage_one_config)
        self.stage_two = TrainingPipeline.from_config(stage_two_config)

    @classmethod
    def from_config(cls, config: TwoStageConfig) -> "UnifiedPipeline":
        return cls(**vars(config))

    def run(self) -> None:
        logger.info("Running induction stage.")
        self.stage_one.run()
        logger.info("Running structured reasoning stage.")
        self.stage_two.run()