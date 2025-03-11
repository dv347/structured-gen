from typing import List

from dataset import TestCase


def evaluate_accuracy(test_cases: List[TestCase]) -> float:
    counter = 0
    for test_case in test_cases:
        if test_case.target == test_case.prediction:
            counter += 1
    return counter / len(test_cases)