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
            "bnf_generation": ("You are an expert programmer, and you need to write a minimally"
                            "sufficient BNF grammar for the given natural language query.\n")
        },
        "exemplar": lambda example: f"Query: {example.source}\nProgram:\n{example.target}\n\n",
        "prediction": {
            "baseline": lambda example: f"Query: {example.source}\nProgram:\n",
            "bnf_generation": lambda example: f"Query: {example.source}\nBNF Grammar:\n"
        }
    }

    @staticmethod
    def from_config(config: PromptConfig) -> "PromptingStrategy":
        classes = {
            ZeroShotConfig: ZeroShot,
            FewShotConfig: FewShot
        }
        config_dict = vars(config)
        config_dict.pop("strategy")
        return classes[type(config)](**config_dict)
    
    @abstractmethod
    def construct_prompt(self, example: Case) -> str:
        raise NotImplementedError("Override me!")
    
    def construct_prompts(self, examples: List[Case]) -> List[str]:
        return [self.construct_prompt(example) for example in examples]
    

class ZeroShot(PromptingStrategy):
    def __init__(self, mode: str):
        self.mode = mode

    def construct_prompt(self, example: Case) -> str:
        return PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.mode] + PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.mode](example)


class FewShot(PromptingStrategy):
    def __init__(self, k: int, exemplars_path: str):
        exemplars = load_from_json(exemplars_path)
        assert k <= len(exemplars), "Few-shot k should be less than or equal to the number of exemplars"
        self.exemplars = exemplars[:k]
        self.mode = "baseline"

    def construct_prompt(self, example: Case) -> str:
        new_exemplars = []
        for exemplar in self.exemplars:
            if exemplar.source != example.source:
                new_exemplars.append(exemplar)
            else:
                logger.info("Found duplicate example in exemplars")
        
        prompt = PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.mode]

        for exemplar in new_exemplars:
            prompt += PromptingStrategy.PROMPT_TEMPLATE["exemplar"](exemplar)

        prompt += PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.mode](example)
        return prompt