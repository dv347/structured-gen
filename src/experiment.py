import re
import time

from cleanup_weights import cleanup_model_dir
from config import ExperimentConfig, ModelConfig, StageConfig
from dataset import Case, TestCase, load_from_json
from llm import LargeLanguageModel
from paths import get_model_dir
from prompt import PromptingStrategy
from results import Results


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        stage_config: StageConfig,
        model: LargeLanguageModel,
        prompting_strategy: PromptingStrategy,
        test_set_path: str,
        cleanup_weights: bool = True # Default is True to avoid hitting storage limits when running multiple experiments, set to False to keep weights
    ):
        self.experiment_name = experiment_name
        self.stage = stage_config.name
        self.grammar_source = stage_config.grammar()
        self.use_embeddings = stage_config.embeddings()
        self.model = model
        self.prompting_strategy = prompting_strategy
        self.test_set_path = test_set_path
        self.cleanup_weights = cleanup_weights

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
        model = LargeLanguageModel.from_config(config.model_config)
        prompting_strategy = PromptingStrategy.from_config(
            config=config.prompt_config, 
            stage_config=config.stage_config,
        )
        return cls(
            experiment_name=config.experiment_name,
            stage_config=config.stage_config,
            model=model,
            prompting_strategy=prompting_strategy,
            test_set_path=config.test_set_path,
            cleanup_weights=config.cleanup_weights
        )
    
    def create_test_case(self, case: Case, prompt: str, prediction: str) -> TestCase:
        target = case.grammar if self.stage == "induction" else case.program
        if self.stage == "baseline_bnf":
            parts = re.split(r'(?i)program:', prediction, maxsplit=1)
            if len(parts) > 1:
                prediction = parts[1].strip()
        return TestCase(source=case.query, target=target, prompt=prompt, prediction=prediction)
    
    def cleanup(self) -> None:
        cleanup_model_dir(get_model_dir(self.model.path))
        if type(self.grammar_source) == ModelConfig:
            cleanup_model_dir(get_model_dir(self.grammar_source.path))
    
    def run(self) -> None:
        start_time = time.time()
        test_set = load_from_json(
            file_path=self.test_set_path, 
            grammar_source=self.grammar_source, 
            use_embeddings=self.use_embeddings
        )
        prompts = self.prompting_strategy.construct_prompts(test_set)
        predictions = self.model.prompt(prompts)
        test_cases = [self.create_test_case(case, prompt, prediction) for case, prompt, prediction in zip(test_set, prompts, predictions)]
        end_time = time.time()
        time_taken = end_time - start_time
        results = Results(self.experiment_name, test_cases, time_taken)
        results.save()
        if self.cleanup_weights:
            self.cleanup()
