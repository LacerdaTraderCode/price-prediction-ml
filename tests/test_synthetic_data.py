"""Tests for synthetic price series generation."""

import numpy as np

from prediction.synthetic_data import generate_price_series


def test_returns_the_requested_number_of_prices():
    prices = generate_price_series(length=100, seed=1)

    assert len(prices) == 100


def test_prices_start_close_to_one_hundred():
    # The series is anchored at 100 before the first day's return is applied,
    # so the first price is 100 scaled by a small, single-day move.
    prices = generate_price_series(length=50, daily_volatility=0.01, seed=1)

    assert 95 < prices[0] < 105


def test_the_same_seed_produces_the_same_series():
    first = generate_price_series(length=200, seed=42)
    second = generate_price_series(length=200, seed=42)

    assert np.array_equal(first, second)


def test_different_seeds_produce_different_series():
    first = generate_price_series(length=200, seed=1)
    second = generate_price_series(length=200, seed=2)

    assert not np.array_equal(first, second)


def test_prices_are_always_positive():
    prices = generate_price_series(length=500, daily_volatility=0.05, seed=7)

    assert (prices > 0).all()


def test_higher_momentum_strength_increases_autocorrelation_of_returns():
    # With momentum, consecutive returns should be positively correlated more
    # often than with none; this is a sanity check on the generator's shape,
    # not a strict guarantee for every seed.
    low_momentum = generate_price_series(
        length=5000, momentum_strength=0.0, daily_volatility=0.01, seed=3
    )
    high_momentum = generate_price_series(
        length=5000, momentum_strength=0.4, daily_volatility=0.01, seed=3
    )

    def lag_one_autocorrelation(prices: np.ndarray) -> float:
        returns = np.diff(prices) / prices[:-1]
        return float(np.corrcoef(returns[:-1], returns[1:])[0, 1])

    assert lag_one_autocorrelation(high_momentum) > lag_one_autocorrelation(low_momentum)
