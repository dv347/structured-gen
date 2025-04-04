import json
import os
from typing import List

from config import ModelConfig
from paths import get_dataset_dir, set_dataset


class Case:
    def __init__(self, query: str, program: str, grammar: str = None, embedding: str = None):
        self.query = query
        program = program.replace("\"\\\"", "\"").replace("\\\"", "")
        self.program = program
        self.grammar = grammar


class TestCase:
    def __init__(self, source: str, target: str, prompt: str, prediction: str):
        self.source = source
        self.target = target
        self.prompt = prompt
        self.prediction = prediction


def load_from_json(
    file_path: str, 
    grammar_source: str | ModelConfig | None = None, 
    use_embeddings: bool = False
) -> List[Case]:
    dataset_dir = get_dataset_dir()
    file_path = os.path.join(dataset_dir, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    cases = [Case(example["query"], example["program"]) for example in data]

    if grammar_source:
        from grammar_loader import GrammarLoader
        loader = GrammarLoader(grammar_source)
        grammars = loader.load_grammars(file_path)
        if use_embeddings:
            embeddings = loader.load_encodings(grammars)
            for case, grammar, embedding in zip(cases, grammars, embeddings):
                case.grammar = grammar
                case.embedding = embedding
        else:
            for case, grammar in zip(cases, grammars):
                case.grammar = grammar

    return cases


def generate_json(stem: str, output_file: str) -> None:
    dataset_dir = get_dataset_dir()
    source_path = os.path.join(dataset_dir, stem + ".src")
    target_path = os.path.join(dataset_dir, stem + ".tgt")
    output_file = os.path.join(dataset_dir, output_file)

    data = []
    with open(source_path, "r", encoding="utf-8") as src, open(target_path, "r", encoding="utf-8") as tgt:
        for query, program in zip(src, tgt):
            data.append({"query": query.strip(), "program": program.strip()})

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump({"data": data}, out, indent=4)