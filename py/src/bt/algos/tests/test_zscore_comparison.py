"""Test comparing z-scores between spread module and backtest engine."""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bt.zscore import calculate_rolling_z
from src.bt.algos.z_model import ZModel


@pytest.fixture
def sample_price_data():
    """Generate sample price data for two correlated symbols."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    base = np.random.randn(100).cumsum() + 100
    s1_prices = base + np.random.randn(100) * 2
    s2_prices = base * 0.5 + np.random.randn(100) * 1 + 50

    df1 = pd.DataFrame(
        {
            "Close": s1_prices,
        },
        index=dates,
    )

    df2 = pd.DataFrame(
        {
            "Close": s2_prices,
        },
        index=dates,
    )

    return df1, df2, dates


def calculate_zscores_spread_module(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    rolling_window: int,
) -> Tuple[pd.Series, List[Dict]]:
    """Calculate z-scores using the spread module approach.

    Returns:
        z_scores: pd.Series with datetime index
        raw_values: List of dicts with timestamp, z, s1, s2 for detailed comparison
    """
    prices1: pd.Series = df1["Close"]
    prices2: pd.Series = df2["Close"]
    dates = df1.index

    z_scores = []
    raw_values = []

    for i in range(len(prices1)):
        s1 = prices1.iloc[: i + 1]
        s2 = prices2.iloc[: i + 1]
        z = calculate_rolling_z(s1, s2, rolling_window)
        z_scores.append(z)
        raw_values.append(
            {
                "timestamp": dates[i],
                "z": z,
                "s1": prices1.iloc[i],
                "s2": prices2.iloc[i],
                "data_points": i + 1,
            }
        )

    z_score_series = pd.Series(z_scores, index=dates)
    return z_score_series, raw_values


def calculate_zscores_bt_engine(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    rolling_window: int,
) -> Tuple[pd.Series, List[Dict]]:
    """Calculate z-scores using the backtest engine approach.

    Simulates tick-by-tick processing like the backtest engine.

    Returns:
        z_scores: pd.Series with datetime index
        raw_values: List of dicts with timestamp, z, s1, s2 for detailed comparison
    """
    symbols = ["s1", "s2"]
    z_model = ZModel(symbols, rolling_window)

    price_buffers: List[dict] = []
    z_scores: List[float] = []
    timestamps: List[pd.Timestamp] = []
    raw_values = []

    for i in range(len(df1)):
        prices = {
            "s1": df1["Close"].iloc[i],
            "s2": df2["Close"].iloc[i],
        }
        price_buffers.append(prices)

        if len(price_buffers) >= rolling_window:
            if len(price_buffers) > rolling_window:
                price_buffers = price_buffers[-rolling_window:]

            z = z_model.calculate_z(price_buffers)
            z_scores.append(z)
            timestamps.append(df1.index[i])
            raw_values.append(
                {
                    "timestamp": df1.index[i],
                    "z": z,
                    "s1": prices["s1"],
                    "s2": prices["s2"],
                    "data_points": len(price_buffers),
                }
            )

    z_score_series = pd.Series(z_scores, index=timestamps)
    return z_score_series, raw_values


def test_zscore_equivalence(sample_price_data):
    """Test that both modules produce identical z-scores."""
    df1, df2, dates = sample_price_data
    rolling_window = 50

    spread_z, spread_raw = calculate_zscores_spread_module(df1, df2, rolling_window)
    bt_z, bt_raw = calculate_zscores_bt_engine(df1, df2, rolling_window)

    min_len = min(len(spread_z), len(bt_z))

    spread_vals = spread_z.iloc[-min_len:].values
    bt_vals = bt_z.iloc[-min_len:].values

    nan_mask = ~(np.isnan(spread_vals) | np.isnan(bt_vals))

    if np.sum(nan_mask) > 0:
        spread_clean = spread_vals[nan_mask]
        bt_clean = bt_vals[nan_mask]

        diff = np.abs(spread_clean - bt_clean)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        assert max_diff < 0.01, (
            f"Max z-score difference: {max_diff:.6f}. "
            f"Spread: {spread_clean[-5:]}, BT: {bt_clean[-5:]}"
        )
        assert mean_diff < 0.001, f"Mean z-score difference: {mean_diff:.6f}"


def test_zscore_timing(sample_price_data):
    """Test that z-scores are calculated at the same timestamps."""
    df1, df2, dates = sample_price_data
    rolling_window = 50

    _, spread_raw = calculate_zscores_spread_module(df1, df2, rolling_window)
    _, bt_raw = calculate_zscores_bt_engine(df1, df2, rolling_window)

    first_spread_z = next((r for r in spread_raw if not np.isnan(r["z"])), None)
    first_bt_z = next((r for r in bt_raw if not np.isnan(r["z"])), None)

    assert first_spread_z is not None, "Spread module should produce z-scores"
    assert first_bt_z is not None, "BT engine should produce z-scores"

    assert first_spread_z["timestamp"] == first_bt_z["timestamp"], (
        f"First z-score at different timestamps: "
        f"spread={first_spread_z['timestamp']}, bt={first_bt_z['timestamp']}"
    )


def test_zscore_nan_handling(sample_price_data):
    """Test that NaN values are handled correctly before window is full."""
    df1, df2, dates = sample_price_data
    rolling_window = 50

    spread_z, _ = calculate_zscores_spread_module(df1, df2, rolling_window)
    bt_z, _ = calculate_zscores_bt_engine(df1, df2, rolling_window)

    spread_nan_count = spread_z.isna().sum()
    bt_nan_count = bt_z.isna().sum()

    expected_nans = rolling_window - 1

    assert spread_nan_count == expected_nans, (
        f"Expected {expected_nans} NaNs in spread z-scores, got {spread_nan_count}"
    )
    assert bt_nan_count == 0, (
        f"BT engine should have 0 NaNs (buffers already full when calculating), got {bt_nan_count}"
    )


# def test_zscore_drift_analysis(sample_price_data):
# """Analyze z-score behavior to detect drift issues.
#
# Returns detailed comparison data for manual inspection.
# """
# df1, df2, dates = sample_price_data
# rolling_window = 50
#
# spread_z, spread_raw = calculate_zscores_spread_module(df1, df2, rolling_window)
# bt_z, bt_raw = calculate_zscores_bt_engine(df1, df2, rolling_window)
#
# comparison_data = []
# for i, (s, b) in enumerate(zip(spread_raw, bt_raw)):
#     comparison_data.append(
#         {
#             "i": i,
#             "spread_z": s["z"],
#             "bt_z": b["z"],
#             "diff": None
#             if np.isnan(s["z"]) or np.isnan(b["z"])
#             else s["z"] - b["z"],
#             "spread_data_points": s["data_points"],
#             "bt_data_points": b["data_points"],
#         }
#     )
#
# spread_valid = spread_z.dropna()
# bt_valid = bt_z.dropna()
#
# if len(spread_valid) > 0 and len(bt_valid) > 0:
#     spread_mean = spread_valid.mean()
#     bt_mean = bt_valid.mean()
#     spread_std = spread_valid.std()
#     bt_std = bt_valid.std()
#
#     stats = {
#         "spread_mean": spread_mean,
#         "bt_mean": bt_mean,
#         "spread_std": spread_std,
#         "bt_std": bt_std,
#         "mean_diff": abs(spread_mean - bt_mean),
#         "spread_min": spread_valid.min(),
#         "spread_max": spread_valid.max(),
#         "bt_min": bt_valid.min(),
#         "bt_max": bt_valid.max(),
#         "spread_range": spread_valid.max() - spread_valid.min(),
#         "bt_range": bt_valid.max() - bt_valid.min(),
#     }
#
#     assert abs(stats["mean_diff"]) < 0.01, (
#         f"Mean z-score drift detected: spread_mean={stats['spread_mean']:.4f}, "
#         f"bt_mean={stats['bt_mean']:.4f}, diff={stats['mean_diff']:.4f}"
#     )
#
#     return stats
#
# return None


# def get_zscore_comparison(
#     s1_prices: List[float],
#     s2_prices: List[float],
#     rolling_window: int,
# ) -> Dict:
# """Get detailed z-score comparison for external use.
#
# Args:
#     s1_prices: List of prices for symbol 1
#     s2_prices: List of prices for symbol 2
#     rolling_window: Rolling window size
#
# Returns:
#     Dict with comparison data including z-scores from both methods,
#     timing info, and statistical comparison
# """
# dates = pd.date_range("2023-01-01", periods=len(s1_prices), freq="D")
# df1 = pd.DataFrame({"Close": s1_prices}, index=dates)
# df2 = pd.DataFrame({"Close": s2_prices}, index=dates)
#
# spread_z, spread_raw = calculate_zscores_spread_module(df1, df2, rolling_window)
# bt_z, bt_raw = calculate_zscores_bt_engine(df1, df2, rolling_window)
#
# comparison = {
#     "spread_z": spread_z,
#     "bt_z": bt_z,
#     "spread_raw": spread_raw,
#     "bt_raw": bt_raw,
#     "are_equal": False,
#     "mean_diff": None,
#     "max_diff": None,
#     "nan_count_spread": int(spread_z.isna().sum()),
#     "nan_count_bt": int(bt_z.isna().sum()),
# }
#
# min_len = min(len(spread_z), len(bt_z))
# if min_len > 0:
#     spread_vals = spread_z.iloc[-min_len:].values
#     bt_vals = bt_z.iloc[-min_len:].values
#
#     nan_mask = ~(np.isnan(spread_vals) | np.isnan(bt_vals))
#     if np.sum(nan_mask) > 0:
#         comparison["mean_diff"] = float(
#             np.mean(np.abs(spread_vals[nan_mask] - bt_vals[nan_mask]))
#         )
#         comparison["max_diff"] = float(
#             np.max(np.abs(spread_vals[nan_mask] - bt_vals[nan_mask]))
#         )
#         comparison["are_equal"] = comparison["max_diff"] < 0.01
#
# return comparison
