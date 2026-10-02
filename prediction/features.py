"""Turns a raw price series into model-ready features and a direction label.

Every feature at row i is computed from data available up to and including
day i; the label is whether day i+1 closes higher than day i. Keeping that
boundary strict is what keeps the dataset free of lookahead bias — a model
that could see tomorrow's price while training would look far better here
than it could ever perform in practice.
"""

import pandas as pd


def build_feature_frame(prices) -> pd.DataFrame:
    series = pd.Series(prices, name="price")
    returns = series.pct_change()

    frame = pd.DataFrame(
        {
            "return_lag_1": returns,
            "return_lag_2": returns.shift(1),
            "return_lag_3": returns.shift(2),
            "price_vs_moving_average_5": series / series.rolling(5).mean() - 1,
            "volatility_5": returns.rolling(5).std(),
            "momentum_10": series / series.shift(10) - 1,
        }
    )
    frame["label_up"] = (series.shift(-1) > series).astype(int)

    return frame.dropna()
