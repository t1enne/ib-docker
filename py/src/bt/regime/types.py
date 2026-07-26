"""Protocol and types for pluggable regime detection."""

from typing import Literal, Protocol

import pandas as pd

RegimeLabel = Literal["BULL", "BEAR", "RANGE"]


class RegimeDetector(Protocol):
    """Pluggable regime detector.

    Takes a DataFrame of prices (multi-symbol candles, columns = "close"/etc.)
    and returns a Series of integer regime labels per timestamp.

    0 = RANGE, 1 = BULL, 2 = BEAR (convention matching HMM regime labels).
    """

    def __call__(self, prices: pd.DataFrame) -> pd.Series: ...


# Canonical mapping: integer label -> string regime
REGIME_INT_TO_LABEL: dict[int, RegimeLabel] = {
    0: "RANGE",
    1: "BULL",
    2: "BEAR",
}

REGIME_LABEL_TO_INT: dict[RegimeLabel, int] = {
    "RANGE": 0,
    "BULL": 1,
    "BEAR": 2,
}
