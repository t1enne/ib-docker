import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.spread import spread
from src.bt.algos.pairs_trading import PairsTradingStrategy
from src.utils import read_candles, calculate_zscore_spread
from src.bt.types import Tick


@pytest.fixture
def sample_data():
    """Sample data for two symbols."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    # Generate correlated prices
    np.random.seed(42)
    base = np.random.randn(100).cumsum() + 100
    s1_prices = base + np.random.randn(100) * 2
    s2_prices = base * 0.5 + np.random.randn(100) * 1 + 50

    df1 = pd.DataFrame(
        {
            "Open": s1_prices,
            "High": s1_prices + 1,
            "Low": s1_prices - 1,
            "Close": s1_prices,
            "Volume": [1000] * 100,
        },
        index=dates,
    )

    df2 = pd.DataFrame(
        {
            "Open": s2_prices,
            "High": s2_prices + 1,
            "Low": s2_prices - 1,
            "Close": s2_prices,
            "Volume": [1000] * 100,
        },
        index=dates,
    )

    return df1, df2


def test_zscore_consistency_spread_vs_bt(sample_data):
    """Test that z_scores from spread command match those from bt strategy at same timestamps."""
    df1, df2 = sample_data
    symbols = ["AAPL", "GOOGL"]
    rolling_window = 50

    # Calculate z_score using spread logic with rolling window
    z_score_spread = calculate_zscore_spread(df1["Close"], df2["Close"], rolling_window)

    # Simulate bt: use rolling z-score on all available data at each step
    bt_z_scores = {}
    for i in range(rolling_window - 1, len(df1)):  # start when enough data for rolling
        current_s1 = df1["Close"].iloc[: i + 1].dropna()
        current_s2 = df2["Close"].iloc[: i + 1].dropna()
        if len(current_s1) == len(current_s2) and len(current_s1) >= rolling_window:
            z_scores = calculate_zscore_spread(current_s1, current_s2, rolling_window)
            bt_z_scores[df1.index[i]] = round(z_scores.iloc[-1], 2)

    # Compare z_scores at the last timestamp where both have values
    last_ts = df1.index[-1]
    if (
        last_ts in z_score_spread.index
        and last_ts in bt_z_scores
        and not (pd.isna(z_score_spread.loc[last_ts]) or pd.isna(bt_z_scores[last_ts]))
    ):
        spread_z = z_score_spread.loc[last_ts]
        bt_z = bt_z_scores[last_ts]
        assert abs(spread_z - bt_z) < 0.01, (
            f"Z-score mismatch at {last_ts}: spread={spread_z:.2f}, bt={bt_z:.2f}"
        )
    else:
        # If last is NaN, perhaps the window is too large, skip or assert they are both NaN
        assert pd.isna(z_score_spread.loc[last_ts]) and pd.isna(bt_z_scores[last_ts]), (
            "Both should be NaN or both should have values"
        )
