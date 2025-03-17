import json
import os
from typing import List

import torch

from config import ModelConfig
from dataset import load_from_json
from grammar_generator import GrammarGenerator
from llm import LargeLanguageModel
from logger import logger
from paths import MODELS_DIR
from prompt import ZeroShot


class GrammarLoader:
    def __init__(self, grammar_source: str | ModelConfig):
        self.is_llm = isinstance(grammar_source, ModelConfig)
        if self.is_llm:
            self.model_config = grammar_source
        else:
            self.variant = grammar_source
            self.generator = GrammarGenerator(path="lispress_full_3.lark", variant=self.variant)

    def load_grammar(self, program: str) -> str:
        assert self.variant != "llm"
        return self.generator.generate(program)

    def load_grammars(self, path: str) -> List[str]:
        if self.is_llm and not self.cache_exists(path):
            logger.info(f'No cache found for model at {self.model_config.path}. Generating cache.')
            self.generate_cache(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.mps.is_available():
                torch.mps.empty_cache()
        if self.is_llm:
            logger.info(f'Loading grammars from cache for model at {self.model_config.path}.')
            return self.load_from_cache(path)
        return [self.load_grammar(case.program) for case in load_from_json(path)]
    
    def generate_cache(self, path: str) -> None:
        model = LargeLanguageModel.from_config(self.model_config)
        prompt_strategy = ZeroShot(stage="induction")
        cache_data = {"data": []}
        cases = load_from_json(path)
        prompts = prompt_strategy.construct_prompts(cases)
        predictions = model.prompt(prompts)
        predictions = [{"query": case.query, "grammar": prediction.strip()} for case, prediction in zip(cases, predictions)]
        cache_data["data"] = predictions

        os.makedirs(self.cache_dir(path), exist_ok=True)
        cache_file = os.path.join(self.cache_dir(path), "cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)

        logger.info(f"Cache saved at {cache_file}")

    def cache_dir(self, path) -> str:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(MODELS_DIR, self.model_config.path, "cache", dataset_name)
    
    def cache_exists(self, path: str) -> bool:
        return os.path.exists(os.path.join(self.cache_dir(path), "cache.json"))

    def load_from_cache(self, path) -> List[str]:
        cache_file = os.path.join(self.cache_dir(path), "cache.json")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)["data"]
        return [entry["grammar"] for entry in cache_data]