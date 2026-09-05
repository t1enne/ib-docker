"""Equity-mode sizing in the execution primitives (``size_mode="equity"``).

Regression coverage for Opt-A ``StrategyContext.{long,short}(size_mode=...)``:
default ``"capital"`` sizing is byte-identical to the legacy ``size * initial
capital`` behavior, while ``"equity"`` sizes a ``size`` off the *live* MTM book
so positions compound with PnL instead of drifting down off fixed seed capital.
"""

from dataclasses import replace

import pytest

from src.bt.strategies.dsl import StrategyContext
from src.bt.strategies.tests.test_dsl import _df, _make_ctx

# price for the fixture close array at the sampled candle.
_PRICE = 100.0


def _rng_many():
    """A flat book whose cash already exceeds initial capital (equity grew).

    ``initial_capital`` stays at the seed value; the divergence between the
    ``"capital"`` (seed) and ``"equity"`` (live cash) bases is what the two
    size modes must reproduce.
    """
    df = _df(120)
    ctx = _make_ctx(df, idx=60)
    eq = _higher_equity_ctx(ctx)
    return eq


def _higher_equity_ctx(ctx: StrategyContext, cash: float = 15000.0) -> StrategyContext:
    from src.bt.engine.utils import merge_bt_state

    portfolio = replace(ctx.state.portfolio, cash=cash)
    return StrategyContext(
        merge_bt_state(ctx.state, dict(portfolio=portfolio)),
        ctx.candle,
        ctx.params,
        ctx.ta,
        ctx.symbols,
        interval="1d",
    )


def _seeded_qty(ctx: StrategyContext, size: float, price: float) -> float:
    return round(size * ctx.state.portfolio.initial_capital / price, 4)


def test_default_capital_mode_unchanged_from_legacy():
    # Legacy path: qty = size * initial_capital / price, no dependence on cash.
    ctx = _make_ctx(_df(120), idx=60)
    ctx.long("AAPL", size=0.1)
    assert len(ctx._signals) == 1
    assert ctx._signals[0].qty == pytest.approx(
        _seeded_qty(ctx, 0.1, float(ctx.price("AAPL")))
    )
    # Even after equity diverges from seed (cash grown), default stays seed-base.
    eq_ctx = _higher_equity_ctx(ctx)
    eq_ctx.long("AAPL", size=0.1)
    assert eq_ctx._signals[0].qty == pytest.approx(
        _seeded_qty(eq_ctx, 0.1, float(eq_ctx.price("AAPL")))
    )


def test_equity_mode_scales_from_live_equity():
    ctx = _make_ctx(_df(120), idx=60)
    # seed = 10_000 but live cash/equity = 15_000 -> equity mode is 1.5x larger.
    eq = _higher_equity_ctx(ctx, cash=15000.0)
    assert eq.current_equity() == pytest.approx(15000.0)
    eq.long("AAPL", size=0.1, size_mode="equity")
    sig = eq._signals[0]
    price = float(eq.price("AAPL"))
    expected = round(0.1 * 15000.0 / price, 4)
    assert sig.qty == pytest.approx(expected)
    # sanity: strictly larger than the seed-based legacy amount.
    assert sig.qty > round(0.1 * 10000.0 / price, 4)


def test_equity_mode_is_opt_in_not_default():
    # Default (no size_mode) and explicit "capital" match legacy.
    a = _higher_equity_ctx(_make_ctx(_df(120), idx=60), cash=15000.0)
    a.long("AAPL", size=0.2, size_mode="capital")
    a_default = _higher_equity_ctx(_make_ctx(_df(120), idx=60), cash=15000.0)
    a_default.long("AAPL", size=0.2)
    assert a._signals[0].qty == a_default._signals[0].qty

    # Explicit "equity" differs (larger, since live book > seed).
    b = _higher_equity_ctx(_make_ctx(_df(120), idx=60), cash=15000.0)
    b.long("AAPL", size=0.2, size_mode="equity")
    assert b._signals[0].qty > a._signals[0].qty


def test_equity_mode_applies_to_short_too():
    eq = _higher_equity_ctx(_make_ctx(_df(120), idx=60), cash=15000.0)
    eq.short("AAPL", size=0.1, size_mode="equity")
    sig = eq._signals[0]
    from src.bt.state import ActionType

    assert sig.action == ActionType.short
    assert sig.qty == pytest.approx(round(0.1 * 15000.0 / float(eq.price("AAPL")), 4))


def test_omitted_size_ignores_mode_and_emits_qty_zero():
    # ``size=None`` routes to the engine shared sizing layer (qty=0) regardless.
    eq = _higher_equity_ctx(_make_ctx(_df(120), idx=60), cash=15000.0)
    eq.long("AAPL", size_mode="equity")
    assert len(eq._signals) == 1
    assert eq._signals[0].qty == 0.0
