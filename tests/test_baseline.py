"""Tests for the naive baselines."""

import numpy as np

from prediction.baseline import majority_class_baseline, persistence_baseline


def test_majority_class_baseline_predicts_the_more_common_training_label():
    train_labels = np.array([1, 1, 1, 0])

    predictions = majority_class_baseline(train_labels, test_length=5)

    assert (predictions == 1).all()


def test_majority_class_baseline_can_predict_all_zeros():
    train_labels = np.array([0, 0, 1])

    predictions = majority_class_baseline(train_labels, test_length=3)

    assert (predictions == 0).all()


def test_majority_class_baseline_returns_the_requested_length():
    predictions = majority_class_baseline(np.array([1, 0, 1]), test_length=7)

    assert len(predictions) == 7


def test_persistence_baseline_predicts_up_when_the_last_return_was_positive():
    predictions = persistence_baseline(np.array([0.01, -0.02, 0.0, 0.03]))

    assert list(predictions) == [1, 0, 0, 1]
