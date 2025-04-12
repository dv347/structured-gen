import os

from config import DatasetPaths, InductionConfig, LoraArgs, ModelConfig, StructuredReasoningConfig, TrainingArgs, TrainingConfig, TwoStageConfig, UnifiedConfig
from logger import logger
from pipelines import TrainingPipeline
from utils import clear_gpu_cache


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
        stage_two_training: TrainingArgs,
        seed: int = 42
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
            training_args=stage_one_training,
            seed=seed
        )

        model_path = os.path.join(induction_dir, "merged_model")
        grammar_source = ModelConfig(path=model_path, batch_size=64, assistant_model=None) if stage_config.grammar_source["use_llm"] else initial_grammar
        struct_config = StructuredReasoningConfig(name="structured_reasoning", grammar_source=grammar_source, use_embeddings=stage_config.use_embeddings)
        self.stage_two_config = TrainingConfig(
            stage_config=struct_config,
            model_path=model_path,
            output_dir=output_dir,
            dataset_paths=stage_two_dataset,
            lora_args=stage_two_lora,
            training_args=stage_two_training,
            seed=seed
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
        clear_gpu_cache()

        logger.info("Running structured reasoning stage.")
        UnifiedPipeline.run_stage(self.stage_two_config)