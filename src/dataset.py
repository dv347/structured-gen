import json
import os
from typing import List

from grammar_generator import GrammarGenerator
from paths import DATA_DIR


class Case:
    def __init__(self, source: str, target: str):
        self.source = source
        target = target.replace("\"\\\"", "\"").replace("\\\"", "")
        self.target = target


class TestCase(Case):
    def __init__(self, source: str, target: str, prompt: str, prediction: str):
        super().__init__(source, target)
        self.prompt = prompt
        self.prediction = prediction


def load_from_json(file_path: str, output_key: str = "program") -> List[Case]:
    assert output_key in ["program", "minimal_grammar"], f"Invalid output key: {output_key}"
    file_path = os.path.join(DATA_DIR, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    return [Case(example["query"], example[output_key]) for example in data]


def generate_json(stem: str, output_file: str) -> None:
    source_path = os.path.join(DATA_DIR, stem + ".src")
    target_path = os.path.join(DATA_DIR, stem + ".tgt")
    output_file = os.path.join(DATA_DIR, output_file)

    grammar = GrammarGenerator("lispress_full_3.lark")

    data = []
    with open(source_path, "r", encoding="utf-8") as src, open(target_path, "r", encoding="utf-8") as tgt:
        for query, program in zip(src, tgt):
            data.append({"query": query.strip(), "minimal_grammar": grammar.generate_minimal_grammar(program.strip()), "program": program.strip()})

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump({"data": data}, out, indent=4)


# generate_json("test", "test.json")