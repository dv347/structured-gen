from abc import ABC, abstractmethod
from typing import List

from config import FewShotConfig, ModelConfig, PromptConfig, ZeroShotConfig
from dataset import Case, load_from_json
from logger import logger


class PromptingStrategy(ABC):
    PROMPT_TEMPLATE = {
        "instruction": {
            "baseline": ("You are an expert programmer, and you need to write a program"
                        " for the given natural language query.\n"),
            "baseline_bnf": ("You are an expert programmer, and you need to write a program"
                        " for the given natural language query. First, start by writing a minimal BNF"
                        " grammar sufficient for the task. After writing the grammar, then write the program.\n"),
            "induction": ("You are an expert programmer, and you need to write a minimally"
                         " sufficient BNF grammar for the given natural language query.\n"),
            "structured_reasoning": ("You are an expert programmer, and you need to write a program"
                                    " for the given natural language query. Use the provided BNF grammar"
                                    " to structure your solution.\n")
        },
        "exemplar": {
            "baseline": lambda example: f"Query: {example.query}\nProgram:\n{example.program}\n\n",
            "baseline_bnf": lambda example: f"Query: {example.query}\nBNF Grammar:\n{example.grammar}\nProgram:\n{example.program}\n\n",
        },
        "prediction": {
            "baseline": lambda example: f"Query: {example.query}\nProgram:\n",
            "baseline_bnf": lambda example: f"Query: {example.query}\nBNF Grammar:\n",
            "induction": lambda example: f"Query: {example.query}\nBNF Grammar:\n",
            "structured_reasoning": lambda example: f"Query: {example.query}\nBNF Grammar: {example.grammar}\nProgram:\n"
        }
    }

    @staticmethod
    def from_config(config: PromptConfig, stage: str, grammar_source: str | ModelConfig) -> "PromptingStrategy":
        classes = {
            ZeroShotConfig: ZeroShot,
            FewShotConfig: FewShot
        }
        config_dict = vars(config)
        config_dict.pop("strategy")
        config_dict["stage"] = stage
        cls = classes[type(config)]
        if cls == FewShot:
            config_dict["grammar_source"] = grammar_source
        return cls(**config_dict)
    
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
    def __init__(self, stage, k: int, exemplars_path: str, grammar_source: str | ModelConfig):
        assert stage in ["baseline", "baseline_bnf"], "Few-shot only supports baseline and baseline_bnf stages"
        self.stage = stage
        exemplars = load_from_json(exemplars_path, grammar_source)
        assert k <= len(exemplars), "Few-shot k should be less than or equal to the number of exemplars"
        self.exemplars = exemplars[:k]

    def construct_prompt(self, example: Case) -> str:
        new_exemplars = []
        for exemplar in self.exemplars:
            if exemplar.query != example.query:
                new_exemplars.append(exemplar)
            else:
                logger.info("Found duplicate example in exemplars")
        
        prompt = PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.stage]

        for exemplar in new_exemplars:
            prompt += PromptingStrategy.PROMPT_TEMPLATE["exemplar"][self.stage](exemplar)

        prompt += PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.stage](example)
        return prompt