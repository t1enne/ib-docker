from typing import List
import numpy as np
import pandas as pd
from src.utils import get_ols_fit_model


def calculate_rolling_z(
    s1: pd.Series[float],
    s2: pd.Series[float],
    window: int,
) -> float:
    """Calculate rolling z-score using OLS regression.

    This uses OLS regression over rolling window:
    - Fit: s1 ~ s2 using log prices
    - spread = s1 - (alpha + beta * s2)
    - z = (spread - rolling_mean) / rolling_std

    Args:
        s1: Prices for symbol 1
        s2: Prices for symbol 2
        window: Rolling window size

    Returns:
        Z-score for the last point in the series
    """
    if len(s1) < window or len(s2) < window:
        return float("nan")

    s1_arr = np.array(s1[-window:])
    s2_arr = np.array(s2[-window:])

    model = get_ols_fit_model(s1_arr, s2_arr)
    alpha, beta = model.params

    spread = s1_arr - (alpha + beta * s2_arr)

    mean = np.mean(spread)
    std = np.std(spread, ddof=1)

    current_spread = s1_arr[-1] - (alpha + beta * s2_arr[-1])
    z = (current_spread - mean) / std if std != 0 else 0.0

    return round(float(z), 2)
