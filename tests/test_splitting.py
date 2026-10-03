"""Tests for the time-respecting train/test split."""

import pandas as pd

from prediction.splitting import time_based_split


def test_split_preserves_chronological_order():
    frame = pd.DataFrame({"value": range(100)})

    train, test = time_based_split(frame, test_fraction=0.2)

    assert train["value"].max() < test["value"].min()


def test_split_sizes_match_the_requested_fraction():
    frame = pd.DataFrame({"value": range(100)})

    train, test = time_based_split(frame, test_fraction=0.25)

    assert len(train) == 75
    assert len(test) == 25


def test_no_row_is_duplicated_or_dropped():
    frame = pd.DataFrame({"value": range(37)})

    train, test = time_based_split(frame, test_fraction=0.3)

    assert len(train) + len(test) == len(frame)
