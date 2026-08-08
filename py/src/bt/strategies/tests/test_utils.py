"""Tests for the shared per-trade sizing + SL/TP helpers (strategies/utils.py)."""

import pytest

from src.bt.strategies.utils import sized_qty, sl_tp_from_pct


# ── sized_qty ──────────────────────────────────────────────────────────


def test_sized_qty_basic():
    """Fraction of cash at price."""
    assert sized_qty(cash=100000, position_size=0.25, price=100.0) == pytest.approx(
        250.0
    )
    assert sized_qty(cash=50000, position_size=0.45, price=22.93) == pytest.approx(
        981.2473
    )


def test_sized_qty_rounds_four_dp():
    assert sized_qty(cash=100000, position_size=0.33, price=7.0) == pytest.approx(
        round(33000 / 7.0, 4)
    )


def test_sized_qty_returns_zero_on_edge_inputs():
    assert sized_qty(cash=0.0, position_size=0.5, price=10.0) == 0.0
    assert sized_qty(cash=1000.0, position_size=0.0, price=10.0) == 0.0
    assert sized_qty(cash=1000.0, position_size=0.5, price=0.0) == 0.0
    assert sized_qty(cash=1000.0, position_size=-0.1, price=10.0) == 0.0


# ── sl_tp_from_pct ─────────────────────────────────────────────────────


def test_sl_tp_long():
    sl, tp = sl_tp_from_pct(100.0, stop_loss=0.05, take_profit=0.2, is_long=True)
    assert sl == pytest.approx(95.0)
    assert tp == pytest.approx(120.0)


def test_sl_tp_short():
    sl, tp = sl_tp_from_pct(100.0, stop_loss=0.05, take_profit=0.2, is_long=False)
    assert sl == pytest.approx(105.0)
    assert tp == pytest.approx(80.0)


def test_sl_tp_zero_pct_disables_each_leg():
    sl, tp = sl_tp_from_pct(100.0, stop_loss=0.0, take_profit=0.2, is_long=True)
    assert sl is None
    assert tp == pytest.approx(120.0)

    sl, tp = sl_tp_from_pct(100.0, stop_loss=0.05, take_profit=0.0, is_long=True)
    assert sl == pytest.approx(95.0)
    assert tp is None


def test_sl_tp_all_disabled():
    sl, tp = sl_tp_from_pct(100.0, stop_loss=0.0, take_profit=0.0, is_long=True)
    assert sl is None
    assert tp is None
