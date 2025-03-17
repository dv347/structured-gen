import json
import os
from typing import List

from paths import DATA_DIR


class Case:
    def __init__(self, query: str, program: str, grammar: str = None):
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


def load_from_json(file_path: str) -> List[Case]:
    file_path = os.path.join(DATA_DIR, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    return [Case(example["query"], example["program"]) for example in data]


def generate_json(stem: str, output_file: str) -> None:
    source_path = os.path.join(DATA_DIR, stem + ".src")
    target_path = os.path.join(DATA_DIR, stem + ".tgt")
    output_file = os.path.join(DATA_DIR, output_file)

    data = []
    with open(source_path, "r", encoding="utf-8") as src, open(target_path, "r", encoding="utf-8") as tgt:
        for query, program in zip(src, tgt):
            data.append({"query": query.strip(), "program": program.strip()})

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump({"data": data}, out, indent=4)


# generate_json("test", "test.json")