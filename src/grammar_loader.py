import json
import os
from typing import List

import torch
from tqdm import tqdm

from config import ModelConfig
from dataset import load_from_json
from grammar_encoder import GrammarEncoder
from grammar_generator import GrammarGenerator
from llm import LargeLanguageModel
from logger import logger
from paths import MODELS_DIR
from prompt import ZeroShot
from utils import clear_gpu_cache


class GrammarLoader:
    def __init__(self, grammar_source: str | ModelConfig):
        self.is_llm = isinstance(grammar_source, ModelConfig)
        if self.is_llm:
            self.model_config = grammar_source
        else:
            self.variant = grammar_source
            self.generator = GrammarGenerator.create(path="lispress_full_3.lark", variant=self.variant)

    def load_grammar(self, program: str) -> str:
        assert self.variant != "llm"
        return self.generator.generate(program)

    def load_grammars(self, path: str) -> List[str]:
        if self.is_llm and not self.cache_exists(path):
            logger.info(f'No cache found for model at {self.model_config.path}. Generating cache.')
            self.generate_cache(path)
            clear_gpu_cache()
        if self.is_llm:
            logger.info(f'Loading grammars from cache for model at {self.model_config.path}.')
            return self.load_from_cache(path)
        return [self.load_grammar(case.program) for case in load_from_json(path)]
    
    def _load_encodings(self, grammars: List[str]) -> List[str]:
        encoder = GrammarEncoder().eval()
        batch_size = 32
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(grammars), batch_size), desc="Encoding grammars"):
                batch_grammars = grammars[i:i+batch_size]
                batch_embeddings = encoder(batch_grammars).cpu()
                batch_embeddings = batch_embeddings.tolist()
                rounded_embeddings = [
                    [round(num, 2) for num in embedding]
                    for embedding in batch_embeddings
                ]
                embeddings.extend(rounded_embeddings)
        return embeddings
    
    def load_encodings(self, grammars: List[str]) -> List[str]:
        output = self._load_encodings(grammars)
        clear_gpu_cache()
        return output
    
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