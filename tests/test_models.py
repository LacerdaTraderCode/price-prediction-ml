"""Tests that the models train and predict without error on realistic feature shapes."""

import numpy as np

from prediction.models import build_neural_network, build_random_forest


def _toy_dataset(size: int = 60):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(size, 6))
    labels = (features[:, 0] > 0).astype(int)
    return features, labels


def test_random_forest_trains_and_predicts_binary_labels():
    features, labels = _toy_dataset()
    model = build_random_forest()

    model.fit(features, labels)
    predictions = model.predict(features)

    assert set(predictions).issubset({0, 1})
    assert len(predictions) == len(labels)


def test_neural_network_trains_and_predicts_binary_labels():
    features, labels = _toy_dataset()
    model = build_neural_network()

    model.fit(features, labels)
    predictions = model.predict(features)

    assert set(predictions).issubset({0, 1})
    assert len(predictions) == len(labels)


def test_neural_network_learns_a_clearly_separable_pattern():
    # The label is just the sign of the first feature; a correctly wired
    # network plus scaler should recover this almost perfectly in-sample.
    features, labels = _toy_dataset(size=300)
    model = build_neural_network()

    model.fit(features, labels)
    accuracy = (model.predict(features) == labels).mean()

    assert accuracy > 0.9
