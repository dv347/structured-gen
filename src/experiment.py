import re
import time
from typing import List

from tqdm import tqdm
from config import ExperimentConfig, ModelConfig, StageConfig
from dataset import Case, TestCase, load_from_json
from grammar_loader import GrammarLoader
from llm import LargeLanguageModel
from prompt import PromptingStrategy
from results import Results


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        stage: str,
        grammar_source: str | ModelConfig,
        model: LargeLanguageModel,
        prompting_strategy: PromptingStrategy,
        test_set_path: str
    ):
        self.experiment_name = experiment_name
        self.stage = stage
        self.grammar_source = grammar_source
        self.model = model
        self.prompting_strategy = prompting_strategy
        self.test_set_path = test_set_path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
        model = LargeLanguageModel.from_config(config.model_config)
        stage = config.stage_config.name
        grammar_source = None
        if stage in ["baseline_bnf", "induction", "structured_reasoning"]:
            grammar_source = config.stage_config.grammar_source
        prompting_strategy = PromptingStrategy.from_config(
            config=config.prompt_config, 
            stage=stage, 
            grammar_source=grammar_source
        )
        return cls(
            experiment_name=config.experiment_name,
            stage=stage,
            grammar_source=grammar_source,
            model=model,
            prompting_strategy=prompting_strategy,
            test_set_path=config.test_set_path
        )
    
    def create_test_case(self, case: Case, prompt: str, prediction: str) -> TestCase:
        target = case.grammar if self.stage == "induction" else case.program
        if self.stage == "baseline_bnf":
            parts = re.split(r'(?i)program:', prediction, maxsplit=1)
            if len(parts) > 1:
                prediction = parts[1].strip()
        return TestCase(source=case.query, target=target, prompt=prompt, prediction=prediction)
    
    def run(self) -> None:
        start_time = time.time()
        test_set = load_from_json(file_path=self.test_set_path, grammar_source=self.grammar_source)
        prompts = self.prompting_strategy.construct_prompts(test_set)
        predictions = self.model.prompt(prompts)
        test_cases = [self.create_test_case(case, prompt, prediction) for case, prompt, prediction in zip(test_set, prompts, predictions)]
        end_time = time.time()
        time_taken = end_time - start_time
        results = Results(self.experiment_name, test_cases, time_taken)
        results.save()