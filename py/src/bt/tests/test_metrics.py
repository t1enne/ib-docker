"""Tests for capital utilization helper in metrics."""

from __future__ import annotations

import pytest

from src.bt.metrics import capital_utilization


class _Pt:
    """Minimal EquityPoint stand-in (equity, positions_value)."""

    def __init__(self, equity: float, positions_value: float) -> None:
        self.equity = equity
        self.positions_value = positions_value


def test_capital_utilization_empty() -> None:
    assert capital_utilization(None) == 0.0
    assert capital_utilization(()) == 0.0


def test_capital_utilization_deployed() -> None:
    pts = (
        _Pt(equity=100_000, positions_value=50_000),  # 50%
        _Pt(equity=110_000, positions_value=110_000),  # 100%
        _Pt(equity=90_000, positions_value=0),  # 0%
    )
    result = capital_utilization(pts)
    assert result == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_capital_utilization_skips_nonpositive_equity() -> None:
    pts = (
        _Pt(equity=100_000, positions_value=25_000),  # 25%
        _Pt(equity=0, positions_value=0),  # skipped
        _Pt(equity=-5, positions_value=1),  # skipped
    )
    result = capital_utilization(pts)
    assert result == pytest.approx(0.25)


def test_capital_utilization_all_invalid() -> None:
    pts = (
        _Pt(equity=0, positions_value=0),
        _Pt(equity=-1, positions_value=0),
    )
    assert capital_utilization(pts) == 0.0


def test_capital_utilization_full_investment() -> None:
    pts = tuple(_Pt(equity=50_000, positions_value=50_000) for _ in range(10))
    assert capital_utilization(pts) == pytest.approx(1.0)
