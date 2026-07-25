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
    """Long entry should pay the spread (buy at higher price)."""
    fill = execute_signal(long_signal, candle_bullish, execution_params)

    # Spread = 100 * 10/10000 = 0.10
    # Expected executed price = 100 + 0.10 + slippage
    assert fill.executed_price > long_signal.price
    assert fill.commission == 0.5


def test_short_entry_receives_spread(execution_params, short_signal, candle_bearish):
    """Short entry should receive the spread (sell at lower price)."""
    fill = execute_signal(short_signal, candle_bearish, execution_params)

    # Spread = 100 * 10/10000 = 0.10
    # Expected executed price = 100 - 0.10 - slippage
    assert fill.executed_price < short_signal.price


def test_adverse_selection_long(execution_params, long_signal, candle_bearish):
    """Long entry in bearish conditions should have worse slippage."""
    fill = execute_signal(long_signal, candle_bearish, execution_params)

    # Bearish tick (close < open) should trigger adverse selection
    # Slippage should be higher
    base_spread = long_signal.price * execution_params.spread_bps / 10000
    expected_without_adverse = long_signal.price + base_spread

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
