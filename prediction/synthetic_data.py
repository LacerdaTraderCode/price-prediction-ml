"""Generates a synthetic price series with weak, realistic momentum plus noise.

Real asset returns sit close to a random walk with a faint, noisy
autocorrelation structure. This generator reproduces that shape instead of
either pure noise, which would make the prediction task unlearnable, or
strong deterministic momentum, which would make it unrealistically easy.
"""

import numpy as np


def generate_price_series(
    length: int,
    momentum_strength: float = 0.05,
    daily_volatility: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Return `length` prices starting at 100, each day's return nudged by the previous one."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(length)
    previous_return = 0.0

    for day in range(length):
        noise = rng.normal(loc=0.0, scale=daily_volatility)
        returns[day] = momentum_strength * previous_return + noise
        previous_return = returns[day]

    return 100.0 * np.cumprod(1 + returns)
