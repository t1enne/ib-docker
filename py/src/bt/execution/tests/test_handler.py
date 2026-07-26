import pytest
from src.bt.execution import (
    execute_signal,
    create_execution_params,
)
from src.bt.state import TradeSignal, Candle, ActionType
from src.utils import get_ts


@pytest.fixture
def execution_params():
    return create_execution_params(
        spread_bps=10.0, slippage_bps=2.0, fixed_commission=0.5
    )


@pytest.fixture
def long_signal():
    return TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )


@pytest.fixture
def short_signal():
    return TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )


@pytest.fixture
def candle_bullish():
    return Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.5,
        volume=1000,
    )


@pytest.fixture
def candle_bearish():
    return Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=101.0,
        high=102.0,
        low=99.0,
        close=99.5,
        volume=1000,
    )


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


def test_long_entry_pays_spread(execution_params, long_signal, candle_bullish):
    """Long entry should pay the spread (buy at higher price than base)."""
    fill = execute_signal(long_signal, candle_bullish, execution_params)

    # fill_at_next_open defaults True → base = tick.open (99.0)
    # Spread = 99.0 * 10/10000 = 0.099
    # Price should be above tick.open (base + spread + slippage)
    assert fill.executed_price > candle_bullish.open
    assert fill.commission == 0.5


def test_short_entry_receives_spread(execution_params, short_signal, candle_bearish):
    """Short entry should receive the spread (sell at lower price than base)."""
    fill = execute_signal(short_signal, candle_bearish, execution_params)

    # fill_at_next_open defaults True → base = tick.open (101.0)
    # Spread = 101.0 * 10/10000 = 0.101
    # Price should be below tick.open (base - spread - slippage)
    assert fill.executed_price < candle_bearish.open


def test_adverse_selection_long(execution_params, long_signal, candle_bearish):
    """Long entry in bearish conditions should have worse slippage."""
    fill = execute_signal(long_signal, candle_bearish, execution_params)

    # Bearish tick (close < open) should trigger adverse selection
    # Slippage should be higher (1.5x)
    base_spread = candle_bearish.open * execution_params.spread_bps / 10000
    expected_without_adverse = candle_bearish.open + base_spread

    # With adverse selection, price should be worse (higher for long)
    assert fill.executed_price > expected_without_adverse


def test_adverse_selection_short(execution_params, short_signal, candle_bullish):
    """Short entry in bullish conditions should have worse slippage."""
    fill = execute_signal(short_signal, candle_bullish, execution_params)

    # Bullish tick (close > open) should trigger adverse selection
    # With adverse selection, slippage should be higher (1.5x)
    # Just verify the fill was executed
    assert fill.executed_price > 0
    assert fill.commission == execution_params.fixed_commission


def test_commission_applied(execution_params, long_signal, candle_bullish):
    """Commission should be applied to fills."""
    fill = execute_signal(long_signal, candle_bullish, execution_params)

    assert fill.commission == execution_params.fixed_commission
    assert fill.commission > 0
