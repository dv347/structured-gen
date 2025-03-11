import time

from tqdm import tqdm
from config import ExperimentConfig
from dataset import TestCase, load_from_json
from llm import LargeLanguageModel
from prompt import PromptingStrategy
from results import Results


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        model: LargeLanguageModel,
        prompting_strategy: PromptingStrategy,
        test_set_path: str
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.prompting_strategy = prompting_strategy
        self.test_set_path = test_set_path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
        model = LargeLanguageModel.from_config(config.model_config)
        prompting_strategy = PromptingStrategy.from_config(config.prompt_config)
        return cls(
            experiment_name=config.experiment_name,
            model=model,
            prompting_strategy=prompting_strategy,
            test_set_path=config.test_set_path
        )

    def run(self) -> None:
        start_time = time.time()
        output_key = "minimal_grammar" if self.prompting_strategy.mode == "induction" else "program"
        test_set = load_from_json(file_path=self.test_set_path, output_key=output_key)
        predictions = []
        for case in tqdm(test_set, desc="Generating predictions", unit="case"):
            prompt = self.prompting_strategy.construct_prompt(case)
            response = self.model.prompt(prompt)
            result = TestCase(
                source=case.source, 
                target=case.target, 
                prompt=prompt, 
                prediction=response
            )
            predictions.append(result)
        end_time = time.time()
        time_taken = end_time - start_time
        results = Results(self.experiment_name, predictions, time_taken)
        results.save()
        