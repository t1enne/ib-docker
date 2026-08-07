"""Regression tests for the `shannons_demon` `cash_leg` equity corruption.

Root cause (fixed):
  - `portfolio.pure._open_position` stamped `Position.type = signal.action`.
    A fresh position opened via a `rebalance` action (cash_leg first build)
    therefore became a `rebalance`-typed Position, which
    `calculate_positions_value`/`update_prices` branch on as a *short*.
    As `last_price` climbed past `2 * entry_price`, that value went negative
    and the equity curve bled to impossible negative returns for a long-only
    SPY+cash portfolio.
  - Fix: map the opening action to a real side (`long`/`short`) — never a
    lifecycle action (`rebalance`) — so rebalance-opened positions are valued
    as their true side.

These tests are offline: they feed synthetic daily candles through the real
engine, so they guard the corruption with no IBKR DB dependency.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.bt.engine.backtest import Backtest, candle_generator, run_backtest
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.portfolio.pure import apply_fill
from src.bt.state import ActionType, FillEvent, TradeSignal, create_initial_portfolio
from src.bt.strategies import init_strat
from src.bt.strategies.shannons_demon import reset_global
from src.bt.types import StrategyConfig


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _rising_spy_df(n_days: int = 2200, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic long-only SPY rising ~10%/year over n_days daily bars."""
    idx = pd.date_range("2015-01-01", periods=n_days, freq="D", tz=None)
    growth = 1.10 ** (1.0 / 252.0)
    closes = start_price * (growth ** np.arange(n_days))
    data = {
        ("SPY", "open"): closes / 1.0,
        ("SPY", "high"): closes * 1.01,
        ("SPY", "low"): closes * 0.99,
        ("SPY", "close"): closes,
        ("SPY", "volume"): np.full(n_days, 1000.0),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cash_leg_config() -> StrategyConfig:
    return StrategyConfig(
        name="shannons-cash-leg-test",
        strategy_type="shannons_demon",
        symbols=["SPY"],
        stop_loss=0,
        take_profit=0,
        initial_capital=100000.0,
        position_size=1.0,
        commission=0.05,
        training_start="2015-01-01",
        training_end="2015-12-31",
        trading_start="2016-01-01",
        trading_end="2021-01-01",
        bars=["1d"],
        benchmark_symbols=["SPY"],
        strategy_params={
            "cash_leg": True,
            "target_weights": [0.5, 0.5],
            "drift_tolerance": 0.01,
            "rebalance_frequency": 5,
            "warmup_bars": 5,
            "trend_gate_enabled": False,
        },
    )


def test_cash_leg_long_only_never_negative_positions_value():
    """A long-only cash_leg SPY backtest must keep positions_value >= 0 everywhere."""
    reset_global()
    cfg = _cash_leg_config()
    bt = Backtest(cfg)
    strat_mod = init_strat("shannons_demon")
    df = _rising_spy_df()

    results, state = run_backtest(
        bt,
        candle_generator(df, cfg),
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=strat_mod,
    )

    points = state.portfolio.equity_curve
    assert len(points) > 0, "expected equity points"

    neg = [p for p in points if p.positions_value < 0]
    assert neg == [], (
        f"cash_leg produced {len(neg)} equity points with negative "
        f"positions_value; first: {neg[0] if neg else None}"
    )

    # Sanity: a long-only SPY+cash portfolio over a tripling SPY must not report
    # an impossible large negative total return.
    assert results.pf.total_return > 0
    assert results.pf.total_return < 3.0


def test_rebalance_open_stamps_real_side_not_lifecycle_action():
    """Opening via a `rebalance` signal must yield a `long` Position, not a
    `rebalance`-typed one (which would be valued as a short)."""
    portfolio = create_initial_portfolio(
        initial_capital=10000.0, start_timestamp=_ts("2024-01-01")
    )
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="SPY",
            timestamp=_ts("2024-01-01"),
            price=200.0,
            qty=10.0,  # positive delta on an empty book -> fresh open
        ),
        filled_qty=10.0,
        executed_price=200.0,
        commission=0.5,
        slippage=0.0,
        timestamp=_ts("2024-01-01"),
    )
    new = apply_fill(portfolio, fill)

    pos = new.positions["SPY"][0]
    assert pos.type == ActionType.long, (
        f"fresh rebalance open should be typed long, got {pos.type}"
    )
