"""Test gap detection — pure functions only."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.cli import _find_gaps_48h, _recap


def _make_df(timestamps: list[datetime]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000] * n,
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


def test_find_gaps_empty():
    assert _find_gaps_48h(pd.DataFrame()) == []


def test_find_gaps_no_gap():
    ts = [datetime(2026, 1, 5, i, 0) for i in range(10)]
    assert _find_gaps_48h(_make_df(ts)) == []


def test_find_gaps_48h_plus():
    ts = [datetime(2026, 1, 5, 10, 0, 0), datetime(2026, 1, 7, 10, 0, 1)]
    gaps = _find_gaps_48h(_make_df(ts))
    assert len(gaps) == 1


def test_weekend_gap_filtered():
    ts = [datetime(2026, 1, 9, 21, 0), datetime(2026, 1, 12, 15, 30)]  # Fri → Mon
    assert _find_gaps_48h(_make_df(ts)) == []


def test_recap():
    ts = [datetime(2026, 1, 5, 10, 0), datetime(2026, 1, 5, 11, 0)]
    result = _recap(_make_df(ts), "AAPL")
    assert "2 rows" in result
    assert "AAPL:" in result
