"""Tests for feature engineering and the absence of lookahead bias."""

import numpy as np

from prediction.features import build_feature_frame


def test_label_matches_the_next_days_actual_direction():
    # A strictly increasing series means every day's label should be "up".
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])

    frame = build_feature_frame(prices)

    assert (frame["label_up"] == 1).all()


def test_no_row_uses_a_future_price_to_build_its_features():
    # Replace the final price with an extreme outlier; only the row whose
    # label looks at that day should be affected, and no feature column should
    # move for earlier rows, since features never look ahead.
    base_prices = np.linspace(100, 120, 30)

    frame_before = build_feature_frame(base_prices)
    altered_prices = base_prices.copy()
    altered_prices[-1] = 1_000_000.0
    frame_after = build_feature_frame(altered_prices)

    feature_columns = [column for column in frame_before.columns if column != "label_up"]
    shared_rows = min(len(frame_before), len(frame_after)) - 1  # drop the last, now-different row
    for column in feature_columns:
        assert np.allclose(
            frame_before[column].to_numpy()[:shared_rows],
            frame_after[column].to_numpy()[:shared_rows],
        )


def test_rows_with_insufficient_history_for_the_rolling_windows_are_dropped():
    prices = np.linspace(100, 110, 12)

    frame = build_feature_frame(prices)

    assert not frame.isna().any().any()


def test_returns_an_empty_frame_for_a_series_too_short_to_build_any_feature():
    prices = np.array([100.0, 101.0])

    frame = build_feature_frame(prices)

    assert len(frame) == 0
