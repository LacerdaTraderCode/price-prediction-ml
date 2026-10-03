"""End-to-end test: the full experiment runs and both models beat a coin flip."""

from scripts.run_experiment import run_experiment


def test_every_model_and_baseline_reports_a_result():
    results = run_experiment()

    names = {result.name for result in results}
    assert names == {"majority_baseline", "persistence_baseline", "random_forest", "neural_network"}


def test_both_learned_models_beat_the_majority_baseline():
    results = {result.name: result for result in run_experiment()}

    assert results["random_forest"].accuracy > results["majority_baseline"].accuracy
    assert results["neural_network"].accuracy > results["majority_baseline"].accuracy
