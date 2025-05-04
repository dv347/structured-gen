from abc import ABC, abstractmethod
from typing import List

from config import FewShotConfig, PromptConfig, StageConfig, ZeroShotConfig
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
            "structured_reasoning": {
                "default": ("You are an expert programmer, and you need to write a program"
                           " for the given natural language query. Use the provided BNF grammar"
                           " to structure your solution.\n"),
                "with_embedding": ("You are an expert programmer, and you need to write a program"
                                  " for the given natural language query. Use the provided BNF grammar"
                                  " and embedding to structure your solution.\n")
            }
        },
        "exemplar": {
            "baseline": lambda example: f"### Query: {example.query}\n### Program:\n{example.program}\n\n",
            "baseline_bnf": lambda example: f"### Query: {example.query}\n### BNF Grammar:\n{example.grammar}\n### Program:\n{example.program}\n\n",
        },
        "prediction": {
            "baseline": lambda example: f"### Query: {example.query}\n ### Program: ",
            "baseline_bnf": lambda example: f"### Query: {example.query}\n ### BNF Grammar: ",
            "induction": lambda example: f"### Query: {example.query}\n ### BNF Grammar: ",
            "structured_reasoning": {
                "default": lambda example: f"### Query: {example.query}\n ### BNF Grammar: {example.grammar}\n ### Program: ",
                "with_embedding": lambda example: f"### Query: {example.query}\n ### BNF Grammar: {example.grammar}\n ### Grammar Embedding: {example.embedding}\n ### Program: "
            }
        }
    }

    @staticmethod
    def from_config(
        config: PromptConfig, 
        stage_config: StageConfig
    ) -> "PromptingStrategy":
        classes = {
            ZeroShotConfig: ZeroShot,
            FewShotConfig: FewShot
        }
        config_dict = vars(config)
        config_dict.pop("strategy")
        config_dict["stage_config"] = stage_config
        return classes[type(config)](**config_dict)
    
    @abstractmethod
    def construct_prompt(self, example: Case) -> str:
        raise NotImplementedError("Override me!")
    
    def construct_prompts(self, examples: List[Case]) -> List[str]:
        return [self.construct_prompt(example) for example in examples]
    

class ZeroShot(PromptingStrategy):
    def __init__(self, stage_config: StageConfig, use_instruction: bool = False):
        self.stage = stage_config.name
        self.use_instruction = use_instruction
        self.instruction = PromptingStrategy.PROMPT_TEMPLATE["instruction"][self.stage]
        self.prediction = PromptingStrategy.PROMPT_TEMPLATE["prediction"][self.stage]
        if self.stage == "structured_reasoning":
            variant = "with_embedding" if stage_config.embeddings() else "default"
            self.instruction = self.instruction[variant]
            self.prediction = self.prediction[variant]

    def construct_prompt(self, example: Case) -> str:
        return self.instruction + self.prediction(example).replace(" ### ", "").replace("### ", "") if self.use_instruction else self.prediction(example)


class FewShot(PromptingStrategy):
    def __init__(self, stage_config: StageConfig, k: int, exemplars_path: str):
        self.stage = stage_config.name
        assert self.stage in ["baseline", "baseline_bnf"], "Few-shot only supports baseline and baseline_bnf stages"
        grammar_source = stage_config.grammar()
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