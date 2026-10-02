"""Time-respecting train/test split for the feature frame.

Shuffling before splitting, the default in most scikit-learn examples, would
let the model train on days that come chronologically after some of its test
days. A real predictive model never gets to see the future; a backtest that
allows it does too, and reports an accuracy nothing like what live use would see.
"""

import pandas as pd


def time_based_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - test_fraction))
    return frame.iloc[:split_index], frame.iloc[split_index:]
