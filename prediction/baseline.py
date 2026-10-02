"""Naive baselines that any real model has to beat to be worth using at all."""

import numpy as np


def majority_class_baseline(train_labels: np.ndarray, test_length: int) -> np.ndarray:
    """Always predicts whichever direction was more common in the training set."""
    majority = 1 if train_labels.mean() >= 0.5 else 0
    return np.full(test_length, majority)


def persistence_baseline(test_return_lag_1: np.ndarray) -> np.ndarray:
    """Predicts that tomorrow continues today's direction."""
    return (test_return_lag_1 > 0).astype(int)
