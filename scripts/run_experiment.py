"""CLI: generates data, trains both models, and prints every predictor compared to baselines.

Usage:
    python -m scripts.run_experiment
"""

from prediction.baseline import majority_class_baseline, persistence_baseline
from prediction.evaluation import EvaluationResult, evaluate_predictions
from prediction.features import build_feature_frame
from prediction.models import build_neural_network, build_random_forest
from prediction.splitting import time_based_split
from prediction.synthetic_data import generate_price_series

SERIES_LENGTH = 4000
MOMENTUM_STRENGTH = 0.2
DAILY_VOLATILITY = 0.01
TEST_FRACTION = 0.25
SEED = 42


def run_experiment() -> list[EvaluationResult]:
    prices = generate_price_series(
        length=SERIES_LENGTH,
        momentum_strength=MOMENTUM_STRENGTH,
        daily_volatility=DAILY_VOLATILITY,
        seed=SEED,
    )
    frame = build_feature_frame(prices)
    train, test = time_based_split(frame, test_fraction=TEST_FRACTION)

    feature_columns = [column for column in frame.columns if column != "label_up"]
    train_features, train_labels = train[feature_columns].to_numpy(), train["label_up"].to_numpy()
    test_features, test_labels = test[feature_columns].to_numpy(), test["label_up"].to_numpy()

    random_forest = build_random_forest()
    random_forest.fit(train_features, train_labels)

    neural_network = build_neural_network()
    neural_network.fit(train_features, train_labels)

    return [
        evaluate_predictions(
            "majority_baseline",
            test_labels,
            majority_class_baseline(train_labels, len(test_labels)),
        ),
        evaluate_predictions(
            "persistence_baseline",
            test_labels,
            persistence_baseline(test["return_lag_1"].to_numpy()),
        ),
        evaluate_predictions("random_forest", test_labels, random_forest.predict(test_features)),
        evaluate_predictions("neural_network", test_labels, neural_network.predict(test_features)),
    ]


def main() -> None:
    results = run_experiment()
    print(f"{'model':22s} {'accuracy':>10s} {'precision':>10s} {'recall':>10s}")
    for result in results:
        print(
            f"{result.name:22s} {result.accuracy:10.3f} "
            f"{result.precision:10.3f} {result.recall:10.3f}"
        )


if __name__ == "__main__":
    main()
