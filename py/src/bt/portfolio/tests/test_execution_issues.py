"""Critical execution issues: both legs execute, sizing, SL/TP from config."""

from src.bt.state import (
    TradeSignal,
    ActionType,
    create_initial_portfolio,
    create_execution_params,
    Candle,
)
from src.bt.portfolio.pure import apply_fill
from src.bt.execution.pure import execute_signal
from src.utils import get_ts


def test_both_legs_execute():
    portfolio = create_initial_portfolio(
        initial_capital=100000, start_timestamp=get_ts("2025-01-01")
    )
    params = create_execution_params(spread_bps=0, slippage_bps=0, fixed_commission=0.0)

    tick_spy = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    tick_qqq = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="QQQ",
        open=400.0,
        high=401.0,
        low=399.0,
        close=400.0,
        volume=1_000_000,
    )

    fill_spy = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=100.0,
            stop_loss=475.0,
            take_profit=550.0,
        ),
        tick_spy,
        params,
    )
    fill_qqq = execute_signal(
        TradeSignal(
            action=ActionType.short,
            symbol="QQQ",
            timestamp=get_ts("2025-01-01"),
            price=400.0,
            z_score=3.0,
            qty=120.0,
            stop_loss=420.0,
            take_profit=360.0,
        ),
        tick_qqq,
        params,
    )

    portfolio = apply_fill(apply_fill(portfolio, fill_spy), fill_qqq)
    assert "SPY" in portfolio.positions
    assert "QQQ" in portfolio.positions
    assert len(portfolio.trades) == 2


def test_explicit_qty_sizing():
    """When signal.qty is set, position uses it directly — no config fallback."""
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=get_ts("2025-01-01")
    )
    tick = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    params = create_execution_params(fixed_commission=0.0, spread_bps=0)
    fill = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=10.0,
        ),
        tick,
        params,
    )
    portfolio_after = apply_fill(portfolio, fill)
    spy = portfolio_after.positions["SPY"]
    assert len(spy) == 1
    assert abs(spy[0].qty - 10.0) < 0.01


def test_sl_tp_from_signal():
    """SL/TP are taken directly from signal, not computed."""
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=get_ts("2025-01-01")
    )
    tick = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    params = create_execution_params(fixed_commission=0.0, spread_bps=0)
    fill = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            z_score=3.0,
            qty=10.0,
            stop_loss=480.0,
            take_profit=530.0,
        ),
        tick,
        params,
    )
    portfolio_after = apply_fill(portfolio, fill)
    pos = portfolio_after.positions["SPY"][0]
    assert pos.stop_loss == 480.0
    assert pos.take_profit == 530.0


def test_open_without_qty_is_noop():
    """If signal.qty is 0 or unset, no position is opened."""
    portfolio = create_initial_portfolio(
        initial_capital=10000, start_timestamp=get_ts("2025-01-01")
    )
    tick = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="SPY",
        open=500.0,
        high=501.0,
        low=499.0,
        close=500.0,
        volume=1_000_000,
    )
    params = create_execution_params(fixed_commission=0.0, spread_bps=0)
    fill = execute_signal(
        TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=get_ts("2025-01-01"),
            price=500.0,
            qty=0.0,
        ),
        tick,
        params,
    )
    portfolio_after = apply_fill(portfolio, fill)
    assert "SPY" not in portfolio_after.positions
    assert len(portfolio_after.trades) == 0
