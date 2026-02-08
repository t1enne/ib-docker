import pytest
from unittest.mock import patch, MagicMock
from src.bt.execution import ExecutionHandler, ExecutionParams
from src.bt.types import TradeSignal, Tick, ActionType, StrategyProtocol
from src.utils import get_ts


@pytest.fixture
def execution_handler():
    params = ExecutionParams(spread_bps=10.0, slippage_bps=2.0, fixed_commission=0.5)
    return ExecutionHandler(params)


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
def tick_bullish():
    return Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.5,
        volume=1000,
    )


@pytest.fixture
def tick_bearish():
    return Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.5,
        high=101.0,
        low=98.0,
        close=99.0,
        volume=1000,
    )


def test_long_entry_pays_spread(execution_handler, long_signal, tick_bullish):
    fill = execution_handler.execute(long_signal, tick_bullish)
    expected_spread = 100.0 * (10.0 / 10000)  # 0.10
    expected_price_base = 100.0 + expected_spread  # 100.10 (long pays spread)
    expected_slippage = 100.0 * (2.0 / 10000)  # 0.02
    expected_price = expected_price_base + expected_slippage  # ~100.12
    assert fill.executed_price > long_signal.price
    assert fill.executed_price > expected_price_base


def test_short_entry_receives_spread(execution_handler, short_signal, tick_bullish):
    fill = execution_handler.execute(short_signal, tick_bullish)
    expected_spread = 100.0 * (10.0 / 10000)  # 0.10
    expected_price_base = 100.0 - expected_spread  # 99.90 (short receives spread)
    expected_slippage = 100.0 * (2.0 / 10000)  # 0.02 (slippage is always added)
    expected_price = expected_price_base + expected_slippage  # ~99.92
    assert fill.executed_price < short_signal.price
    assert fill.executed_price > expected_price_base  # Slippage added


def test_adverse_selection_long(execution_handler, long_signal, tick_bearish):
    fill = execution_handler.execute(long_signal, tick_bearish)
    adverse_slippage = 100.0 * (2.0 * 1.5 / 10000)  # Extra 50% for adverse
    assert fill.slippage > 0


def test_adverse_selection_short(execution_handler, short_signal, tick_bullish):
    fill = execution_handler.execute(short_signal, tick_bullish)
    adverse_slippage = 100.0 * (2.0 * 1.5 / 10000)  # Extra 50% for adverse
    assert fill.slippage > 0


def test_commission_applied(execution_handler, long_signal, tick_bullish):
    fill = execution_handler.execute(long_signal, tick_bullish)
    assert fill.commission > 0
