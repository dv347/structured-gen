import json
import os
from tqdm import tqdm
from typing import List

import torch

from dataset import Case, load_from_json
from grammar_generator import GrammarGenerator
from llm import LargeLanguageModel
from logger import logger
from paths import MODELS_DIR
from prompt import ZeroShot


class GrammarLoader:
    def __init__(self, grammar_source: str | dict):
        self.is_llm = isinstance(grammar_source, dict) and "llm" in grammar_source
        if self.is_llm:
            self.model_path = os.path.join(MODELS_DIR, grammar_source["llm"])
            self.generator = None
        else:
            self.variant = grammar_source
            self.generator = GrammarGenerator(path="lispress_full_3.lark", variant=self.variant)
            self.model_path = None

    def load_grammar(self, case: Case) -> str:
        assert self.variant != "llm"
        return self.generator.generate(case.target)

    def load_grammars(self, path: str) -> List[str]:
        if self.model_path and not self.cache_exists(path):
            logger.info(f'No cache found for model at {self.model_path}. Generating cache.')
            self.generate_cache(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.mps.is_available():
                torch.mps.empty_cache()
        if self.model_path:
            return self.load_from_cache(path)
        return [self.load_grammar(case) for case in load_from_json(path)]
    
    def generate_cache(self, path: str) -> None:
        model = LargeLanguageModel(path=self.model_path, assistant_model=None)
        prompt_strategy = ZeroShot(stage="induction")
        cache_data = {"data": []}
        cases = load_from_json(path)
        for case in tqdm(cases, desc="Generating cache", unit="case"):
            prompt = prompt_strategy.construct_prompt(case)
            response = model.prompt(prompt).strip()
            cache_data["data"].append({"query": case.query, "grammar": response})

        os.makedirs(self.cache_dir(path), exist_ok=True)
        cache_file = os.path.join(self.cache_dir(path), "cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)

        logger.info(f"Cache saved at {cache_file}")

    def cache_dir(self, path) -> str:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.model_path, "cache", dataset_name)
    
    def cache_exists(self, path: str) -> bool:
        return os.path.exists(os.path.join(self.cache_dir(path), "cache.json"))

    def load_from_cache(self, path) -> List[str]:
        cache_file = os.path.join(self.cache_dir(path), "cache.json")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)["data"]
        return [entry["grammar"] for entry in cache_data]