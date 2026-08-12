"""Tests for the pure position-sizing layer."""

from src.bt.size.pure import SizingParams, compute_qty, risk_sized_qty, sized_signal
from src.bt.state import TradeSignal, ActionType, Candle
from src.utils import get_ts


def _params(**overrides) -> SizingParams:
    defaults: dict = {
        "sizing_mode": "equity",
        "size": 0.0,
        "max_symbol_allocation": 1.0,
    }
    defaults.update(overrides)
    return SizingParams(**defaults)


def test_equity_compounding():
    # equity grows from 100 -> 1000; same size keeps investing 10% of equity.
    a = compute_qty(equity=100, cash=100, price=1.0, params=_params(size=0.1))
    b = compute_qty(equity=1000, cash=1000, price=1.0, params=_params(size=0.1))
    assert a == 10.0
    assert b == 100.0


def test_cash_mode_uses_cash():
    p = _params(sizing_mode="cash", size=0.5)
    qty = compute_qty(equity=1000, cash=100, price=1.0, params=p)
    assert qty == 50.0  # 50% of available cash (100), not 50% of equity


def test_equity_mode_uses_equity():
    p = _params(sizing_mode="equity", size=0.5)
    qty = compute_qty(equity=1000, cash=1000, price=1.0, params=p)
    assert qty == 500.0


def test_fixed_mode_uses_fixed_nominal_amount():
    p = _params(sizing_mode="fixed", size=200.0)
    qty = compute_qty(equity=1000, cash=500, price=4.0, params=p)
    assert qty == 50.0  # $200 fixed / $4 price


def test_cash_clamp():
    # 60% of 200 equity = 120 -> qty 120 at price 1; only 50 cash available.
    p = _params(sizing_mode="equity", size=0.6)
    qty = compute_qty(equity=200, cash=50, price=1.0, params=p)
    assert qty == 50.0
    assert qty * 1.0 <= 50.0


def test_max_symbol_allocation_cap():
    # size 0.8 of equity, but per-symbol cap 0.3 -> qty*price capped at 0.3*equity.
    p = _params(sizing_mode="equity", size=0.8, max_symbol_allocation=0.3)
    qty = compute_qty(equity=100, cash=100, price=1.0, params=p)
    assert qty == 30.0


def test_invalid_inputs_return_zero():
    p = _params(size=0.1)
    assert compute_qty(equity=100, cash=100, price=0.0, params=p) == 0.0
    assert compute_qty(equity=100, cash=100, price=-5, params=p) == 0.0
    assert compute_qty(equity=100, cash=100, price=1.0, params=_params()) == 0.0
    # size=0


def test_determinism():
    p = _params(size=0.25, max_symbol_allocation=0.4)
    first = compute_qty(equity=500, cash=500, price=2.0, params=p)
    for _ in range(10):
        assert compute_qty(equity=500, cash=500, price=2.0, params=p) == first


def test_from_dict_coercion():
    p = SizingParams.from_dict(
        {
            "sizing_mode": "cash",
            "size": 0.2,
            "risk_pct": 0.01,  # ignored: risk sizing is strategy-owned now
            "max_symbol_allocation": 0.5,
        }
    )
    assert p.sizing_mode == "cash"
    assert p.size == 0.2
    assert p.max_symbol_allocation == 0.5
    assert not hasattr(p, "risk_pct")

    p_bad = SizingParams.from_dict({"sizing_mode": "bogus"})
    assert p_bad.sizing_mode == "equity"


def test_sized_signal_fills_when_qty_zero():
    p = _params(size=0.1)
    sig = TradeSignal(
        action=ActionType.long,
        symbol="SPY",
        timestamp=get_ts("2025-01-01"),
        price=10.0,
        qty=0.0,
    )
    candle = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        volume=1000,
    )
    sized = sized_signal(sig, equity=100, cash=100, candle=candle, params=p)
    assert sized.qty == 1.0


def test_sized_signal_leaves_explicit_qty_untouched():
    p = _params(size=0.1)
    sig = TradeSignal(
        action=ActionType.long,
        symbol="SPY",
        timestamp=get_ts("2025-01-01"),
        price=10.0,
        qty=7.0,
    )
    candle = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        volume=1000,
    )
    sized = sized_signal(sig, equity=100, cash=100, candle=candle, params=p)
    assert sized.qty == 7.0


# ---------------------------------------------------------------------------
# risk-targeted sizing (strategy-owned helper)
# ---------------------------------------------------------------------------


def test_risk_sized_qty_long():
    # qty = equity * risk_pct / stop_dist = 100 * 0.01 / 0.50 = 2.0
    assert risk_sized_qty(equity=100, price=10.0, stop_dist=0.50, risk_pct=0.01) == 2.0


def test_risk_sized_qty_atr_distance():
    # stop distance from ATR*mult, e.g. ATR 1.00 * mult 1.5 = 1.5
    assert risk_sized_qty(equity=300, price=10.0, stop_dist=1.5, risk_pct=0.01) == 2.0


def test_risk_sized_qty_invalid_inputs_return_zero():
    assert risk_sized_qty(equity=0, price=10.0, stop_dist=1.0, risk_pct=0.01) == 0.0
    assert risk_sized_qty(equity=100, price=0.0, stop_dist=1.0, risk_pct=0.01) == 0.0
    assert risk_sized_qty(equity=100, price=10.0, stop_dist=0.0, risk_pct=0.01) == 0.0
    assert risk_sized_qty(equity=100, price=10.0, stop_dist=-1.0, risk_pct=0.01) == 0.0
    assert risk_sized_qty(equity=100, price=10.0, stop_dist=1.0, risk_pct=0.0) == 0.0


def test_risk_sized_qty_deterministic():
    first = risk_sized_qty(equity=500, price=2.0, stop_dist=0.25, risk_pct=0.02)
    for _ in range(10):
        assert (
            risk_sized_qty(equity=500, price=2.0, stop_dist=0.25, risk_pct=0.02)
            == first
        )
