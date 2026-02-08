from typing import List
import numpy as np


def calculate_rolling_z(
    s1: List[float],
    s2: List[float],
    window: int,
) -> float:
    """Calculate rolling z-score using rolling beta and rolling mean/std.

    This matches the spread module behavior:
    - beta = Cov(s1, s2) / Var(s2) computed over rolling window
    - spread = s1 - beta * s2
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

    cov = np.cov(s1_arr, s2_arr)[0, 1]
    var = np.var(s2_arr, ddof=1)
    beta = cov / var if var != 0 else 1.0

    spreads = s1_arr - beta * s2_arr

    mean = np.mean(spreads)
    std = np.std(spreads, ddof=1)

    spread = s1[-1] - beta * s2[-1]
    z = (spread - mean) / std if std != 0 else 0.0

    return round(float(z), 2)
