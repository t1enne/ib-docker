"""Protocol and types for pluggable regime detection.

Two separate concepts:

  TrendRegime  — market direction (BULL/BEAR/RANGE)
  VolRegime    — volatility level (LOW_VOL/MED_VOL/HIGH_VOL)

Detectors implement RegimeDetector: (prices) -> pd.Series[int].
The mapping from int to label depends on which concept the detector models.
"""

from __future__ import annotations

from typing import Literal, Protocol

import pandas as pd

# ---------------------------------------------------------------------------
# Trend regime (directional)
# ---------------------------------------------------------------------------

TrendRegime = Literal["BULL", "BEAR", "RANGE"]

TREND_INT_TO_LABEL: dict[int, TrendRegime] = {
    0: "RANGE",
    1: "BULL",
    2: "BEAR",
}

TREND_LABEL_TO_INT: dict[TrendRegime, int] = {
    "RANGE": 0,
    "BULL": 1,
    "BEAR": 2,
}


# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------

VolRegime = Literal["LOW_VOL", "MED_VOL", "HIGH_VOL"]

VOL_INT_TO_LABEL: dict[int, VolRegime] = {
    0: "LOW_VOL",
    1: "MED_VOL",
    2: "HIGH_VOL",
}

VOL_LABEL_TO_INT: dict[VolRegime, int] = {
    "LOW_VOL": 0,
    "MED_VOL": 1,
    "HIGH_VOL": 2,
}


# ---------------------------------------------------------------------------
# Combined regime: what the model updater stores on ModelState
# ---------------------------------------------------------------------------


class RegimeDetector(Protocol):
    """Pluggable regime detector.

    Takes a DataFrame of prices (multi-symbol candles, columns = "close"/etc.)
    and returns a Series of integer regime labels per timestamp.

    Interpretation depends on the detector implementation:
      - Trend detectors: 0=RANGE, 1=BULL, 2=BEAR
      - Vol detectors:   0=LOW_VOL, 1=MED_VOL, 2=HIGH_VOL
    """

    def __call__(self, prices: pd.DataFrame) -> pd.Series: ...
