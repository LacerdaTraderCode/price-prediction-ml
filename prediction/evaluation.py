"""Computes and compares accuracy, precision, and recall for each candidate predictor."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    accuracy: float
    precision: float
    recall: float


def evaluate_predictions(
    name: str, true_labels: np.ndarray, predicted_labels: np.ndarray
) -> EvaluationResult:
    return EvaluationResult(
        name=name,
        accuracy=accuracy_score(true_labels, predicted_labels),
        precision=precision_score(true_labels, predicted_labels, zero_division=0),
        recall=recall_score(true_labels, predicted_labels, zero_division=0),
    )
