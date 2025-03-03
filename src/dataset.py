import json
import os
from typing import List, Tuple

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


def load_examples(stem: str) -> List[Case]:  
    source_path = stem + ".src"
    target_path = stem + ".tgt"

    with open(source_path, "r", encoding="utf-8") as src_file:
        source_lines = src_file.readlines()
    with open(target_path, "r", encoding="utf-8") as tgt_file:
        target_lines = tgt_file.readlines()

    assert len(source_lines) == len(target_lines)

    examples = [Case(source=s.strip(), target=t.strip()) for s, t in zip(source_lines, target_lines)]
    return examples


def load_data() -> Tuple[List[Case], List[Case], List[Case]]:
    train_examples = load_examples(os.path.join(DATA_DIR, "train"))
    dev_examples = load_examples(os.path.join(DATA_DIR, "valid"))
    test_examples = load_examples(os.path.join(DATA_DIR, "test"))

    return train_examples, dev_examples, test_examples


def load_from_json(file_path: str) -> List[Case]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    return [Case(example["query"], example["program"]) for example in data]


def generate_json(stem: str, output_file: str) -> None:
    source_path = stem + ".src"
    target_path = stem + ".tgt"

    data = []
    with open(source_path, "r", encoding="utf-8") as src, open(target_path, "r", encoding="utf-8") as tgt:
        for query, program in zip(src, tgt):
            data.append({"query": query.strip(), "program": program.strip()})

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump({"data": data}, out, indent=4)


# generate_json(os.path.join(DATA_DIR, "test"), os.path.join(DATA_DIR, "test.json"))