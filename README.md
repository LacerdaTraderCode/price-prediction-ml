# Price Prediction with ML

A next-day price-direction classifier, comparing a classical scikit-learn ensemble and a small neural network against two naive baselines on a synthetic price series with known, tunable momentum.

This is a methodology exercise, not a trading signal: the price data is generated, not real, specifically so the question being asked — "can these models recover a signal that is actually there?" — has a knowable answer. Nothing here should be read as investment advice or a claim about real markets.

## What it does

1. `prediction/synthetic_data.py` generates a price series with weak, tunable momentum plus noise — close to a random walk, not a deterministic pattern.
2. `prediction/features.py` builds lagged returns, a moving-average ratio, rolling volatility, and 10-day momentum as features, with a strict cutoff so no feature ever sees a future price.
3. `prediction/splitting.py` splits train and test chronologically — no shuffling, so the model never trains on data from after its test period.
4. `prediction/models.py` trains a `RandomForestClassifier` and an `MLPClassifier` (a real neural network, via scikit-learn rather than PyTorch/Keras — see [`docs/architecture.md`](docs/architecture.md) for why) on next-day direction.
5. `prediction/baseline.py` and `prediction/evaluation.py` compare both models against a majority-class baseline and a "today's direction continues" baseline.

Run it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.run_experiment
```

```
model                    accuracy  precision     recall
majority_baseline           0.499      0.000      0.000
persistence_baseline        0.588      0.589      0.590
random_forest               0.565      0.564      0.580
neural_network              0.555      0.554      0.572
```

Both learned models clearly beat the majority-class baseline — they are recovering real signal, not noise — but neither beats the two-line persistence baseline. That is the honest result for this data: the synthetic series is generated directly from yesterday's return, so a model with extra features has more ways to pick up noise alongside that one real signal, not more signal to find. See [`docs/architecture.md`](docs/architecture.md) for why that comparison is left in rather than tuned away.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite checks the generator's reproducibility, that no feature can see a future price, that the split stays chronological, both models train and predict correctly, and that the full experiment reliably beats the majority baseline end to end.

## License

MIT — see [LICENSE](LICENSE).
