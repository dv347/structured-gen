from abc import ABC, abstractmethod
from typing import List

from config import FewShotConfig, PromptConfig, ZeroShotConfig
from dataset import Case, load_from_json
from logger import logger


class PromptingStrategy(ABC):
    PROMPT_TEMPLATE = {
        "instruction": {
            "baseline": ("You are an expert programmer, and you need to write a program"
                        " for the given natural language query.\n"),
            "induction": ("You are an expert programmer, and you need to write a minimally"
                         " sufficient BNF grammar for the given natural language query.\n"),
            "structured_reasoning": ("You are an expert programmer, and you need to write a program"
                                    " for the given natural language query. Use the provided BNF grammar"
                                    " to structure your solution.\n")
        },
        "exemplar": lambda example: f"Query: {example.query}\nProgram:\n{example.program}\n\n",
        "prediction": {
            "baseline": lambda example: f"Query: {example.query}\nProgram:\n",
            "induction": lambda example: f"Query: {example.query}\nBNF Grammar:\n",
            "structured_reasoning": lambda example: f"Query: {example.query}\nBNF Grammar: {example.grammar}\nProgram:\n"
        }
    }

    @staticmethod
    def from_config(config: PromptConfig, stage: str) -> "PromptingStrategy":
        classes = {
            ZeroShotConfig: ZeroShot,
            FewShotConfig: FewShot
        }
        config_dict = vars(config)
        config_dict.pop("strategy")
        if type(config) == ZeroShotConfig:
            config_dict["stage"] = stage
        return classes[type(config)](**config_dict)
    
    @abstractmethod
    def construct_prompt(self, example: Case) -> str:
        raise NotImplementedError("Override me!")
    
    def construct_prompts(self, examples: List[Case]) -> List[str]:
        return [self.construct_prompt(example) for example in examples]
    

class ZeroShot(PromptingStrategy):
    def __init__(self, stage: str):
        self.stage = stage

    def construct_prompt(self, example: Case) -> str:
        return PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.stage] + PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.stage](example)


class FewShot(PromptingStrategy):
    def __init__(self, k: int, exemplars_path: str):
        exemplars = load_from_json(exemplars_path)
        assert k <= len(exemplars), "Few-shot k should be less than or equal to the number of exemplars"
        self.exemplars = exemplars[:k]
        self.stage = "baseline" # Few-shot only supports baseline stage

    def construct_prompt(self, example: Case) -> str:
        new_exemplars = []
        for exemplar in self.exemplars:
            if exemplar.source != example.source:
                new_exemplars.append(exemplar)
            else:
                logger.info("Found duplicate example in exemplars")
        
        prompt = PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.stage]

        for exemplar in new_exemplars:
            prompt += PromptingStrategy.PROMPT_TEMPLATE["exemplar"](exemplar)

        prompt += PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.stage](example)
        return prompt