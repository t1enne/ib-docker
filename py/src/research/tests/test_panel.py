"""Unit tests for pure panel window selection (no DB needed)."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.research.panel import MemberSpan, select_full_history


def _ts(v: str) -> pd.Timestamp:
    """Parse a YYYY-MM-DD literal into a Timestamp (runtime-narrowed)."""
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _shift(base: pd.Timestamp, days: int) -> pd.Timestamp:
    ts = base + timedelta(days=days)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _span(ticker: str, days_after_epoch: int, n_rows: int) -> MemberSpan:
    founding = _ts("2019-11-01")
    start = _shift(founding, days_after_epoch)
    end = _shift(start, 2500)  # uniform long coverage
    return MemberSpan(ticker, start, end, n_rows)


def test_founding_cohort_kept_late_start_dropped() -> None:
    """Members starting near the founding date are kept; a very late IPO is not."""
    offsets = {"ALNY": 0, "AMGN": 8, "BIIB": 0, "VRTX": 20, "LLY": 0, "REGN": 20}
    spans = {t: _span(t, d, 900) for t, d in offsets.items()}
    # A member whose history begins ~7 years late (recent IPO / partial sync).
    spans["AZN"] = _span("AZN", days_after_epoch=2400, n_rows=700)
    kept, dropped = select_full_history(spans)
    assert "AZN" in dropped
    assert "ALNY" in kept and "VRTX" in kept
    assert len(kept) == len(offsets)


def test_below_min_rows_is_dropped() -> None:
    """A founding-date member with too few rows is treated as unusable."""
    spans = {t: _span(t, d, 900) for t, d in {"A": 0, "B": 1, "C": 2}.items()}
    spans["TINY"] = MemberSpan(
        "TINY", _ts("2019-11-01"), _shift(_ts("2019-11-01"), 2500), 40
    )
    kept, dropped = select_full_history(spans, min_rows=50)
    assert set(kept) == {"A", "B", "C"}
    assert "TINY" in dropped


def test_single_member_no_drop() -> None:
    kept, dropped = select_full_history({"SOLO": _span("SOLO", 0, 900)})
    assert kept == ("SOLO",)
    assert dropped == ()


def test_two_late_uniform_members_kept() -> None:
    """A pair of uniformly-late members is still coherent and both are kept."""
    a = _span("U", days_after_epoch=2000, n_rows=450)
    b = MemberSpan("V", a.start, _shift(a.start, 2500), 450)
    kept, dropped = select_full_history({"U": a, "V": b})
    assert set(kept) == {"U", "V"}
    assert dropped == ()
