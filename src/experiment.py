import time
from typing import List

from tqdm import tqdm
from config import ExperimentConfig, StageConfig
from dataset import Case, TestCase, load_from_json
from grammar_loader import GrammarLoader
from llm import LargeLanguageModel
from prompt import PromptingStrategy
from results import Results


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        stage_config: StageConfig,
        model: LargeLanguageModel,
        prompting_strategy: PromptingStrategy,
        test_set_path: str
    ):
        self.experiment_name = experiment_name
        self.stage = stage_config.name
        self.stage_config = stage_config
        self.model = model
        self.prompting_strategy = prompting_strategy
        self.test_set_path = test_set_path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
        model = LargeLanguageModel.from_config(config.model_config)
        prompting_strategy = PromptingStrategy.from_config(config.prompt_config, config.stage_config.name)
        return cls(
            experiment_name=config.experiment_name,
            stage_config=config.stage_config,
            model=model,
            prompting_strategy=prompting_strategy,
            test_set_path=config.test_set_path
        )
    
    def add_grammars(self, cases: List[Case]) -> List[Case]:
        loader = GrammarLoader(grammar_source=self.stage_config.grammar_source)
        grammars = loader.load_grammars(self.test_set_path)
        for case, grammar in zip(cases, grammars):
            case.grammar = grammar
        return cases

    def run(self) -> None:
        start_time = time.time()
        test_set = load_from_json(file_path=self.test_set_path)
        if self.stage in ["induction", "structured_reasoning"]:
            test_set = self.add_grammars(test_set)
        predictions = []
        for case in tqdm(test_set, desc="Generating predictions", unit="case"):
            prompt = self.prompting_strategy.construct_prompt(case)
            response = self.model.prompt(prompt)
            target = case.grammar if self.stage == "induction" else case.program
            result = TestCase(
                source=case.query, 
                target=target, 
                prompt=prompt, 
                prediction=response
            )
            predictions.append(result)
        end_time = time.time()
        time_taken = end_time - start_time
        results = Results(self.experiment_name, predictions, time_taken)
        results.save()
        