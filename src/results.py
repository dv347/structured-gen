import json
import os
from typing import List

from dataset import TestCase
from eval import evaluate_accuracy
from paths import RESULTS_DIR


class Results:
    def __init__(self, experiment_name: str, cases: List[TestCase], time_taken: float):
        self.cases = cases
        self.accuracy = evaluate_accuracy(cases)
        self.time_taken = time_taken
        self.dir = os.path.join(RESULTS_DIR, experiment_name)
        os.makedirs(self.dir, exist_ok=True)

    def save(self) -> None:
        predictions_file = os.path.join(self.dir, "predictions.json")
        predictions_data = [
            {
                "source": case.source,
                "prompt": case.prompt,
                "y_true": case.target,
                "y_pred": case.prediction
            }
            for case in self.cases
        ]
        
        os.makedirs(self.dir, exist_ok=True)
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=4)

        results_file = os.path.join(self.dir, "results.json")
        results_data = {
            "accuracy": self.accuracy,
            "time_taken (seconds)": round(self.time_taken),
            "time_taken (minutes)": round(self.time_taken / 60),
            "time per example (seconds)": round(self.time_taken / len(self.cases), 2),
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4)