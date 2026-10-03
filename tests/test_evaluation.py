"""Tests for the evaluation metrics wrapper."""

import numpy as np

from prediction.evaluation import evaluate_predictions


def test_perfect_predictions_score_one_on_every_metric():
    labels = np.array([1, 0, 1, 1, 0])

    result = evaluate_predictions("perfect", labels, labels)

    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_accuracy_reflects_the_fraction_of_correct_predictions():
    true_labels = np.array([1, 0, 1, 0])
    predicted = np.array([1, 0, 0, 0])

    result = evaluate_predictions("partial", true_labels, predicted)

    assert result.accuracy == 0.75


def test_predicting_no_positives_does_not_raise_on_undefined_precision():
    true_labels = np.array([0, 0, 0])
    predicted = np.array([0, 0, 0])

    result = evaluate_predictions("all_negative", true_labels, predicted)

    assert result.precision == 0.0
    assert result.recall == 0.0


def test_the_name_is_carried_into_the_result():
    labels = np.array([1, 0])

    result = evaluate_predictions("my_model", labels, labels)

    assert result.name == "my_model"
