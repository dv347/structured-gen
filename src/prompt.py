from abc import ABC, abstractmethod
import os
from typing import List

from config import FewShotConfig, PromptConfig, ZeroShotConfig
from dataset import Case, load_from_json
from logger import logger
from paths import DATA_DIR


class PromptingStrategy(ABC):
    def __init__(self):
        self.prompt_template = {
            "instruction": ("You are an expert programmer, and you need to write a program" 
                            " for the given natural language query.\n"),
            "exemplar": lambda example: f"query: {example.source}\nprogram:\n{example.target}\n\n",
            "prediction": lambda example: f"query: {example.source}\nprogram:\n",
        }

    @staticmethod
    def from_config(config: PromptConfig) -> "PromptingStrategy":
        if type(config) == ZeroShotConfig:
            return ZeroShot()
        elif type(config) == FewShotConfig:
            return FewShot(config.exemplars_path)
        else:
            raise ValueError(f"Unsupported prompt config: {config}")

    @abstractmethod
    def construct_prompt(self, example: Case) -> str:
        raise NotImplementedError("Override me!")
    
    def construct_prompts(self, examples: List[Case]) -> List[str]:
        return [self.construct_prompt(example) for example in examples]
    

class ZeroShot(PromptingStrategy):
    def construct_prompt(self, example: Case) -> str:
        return self.prompt_template["instruction"] + self.prompt_template["prediction"](example)


class FewShot(PromptingStrategy):
    def __init__(self, exemplars_path: str):
        super().__init__()
        self.exemplars = load_from_json(exemplars_path)

    def construct_prompt(self, example: Case) -> str:
        new_exemplars = []
        for exemplar in self.exemplars:
            if exemplar.source != example.source:
                new_exemplars.append(exemplar)
            else:
                logger.info("Found duplicate example in exemplars")
        
        prompt = self.prompt_template["instruction"]

        for exemplar in new_exemplars:
            prompt += self.prompt_template["exemplar"](exemplar)

        prompt += self.prompt_template["prediction"](example)
        return prompt