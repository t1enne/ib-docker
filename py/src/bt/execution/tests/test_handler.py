"""Tests for execution handler — critical path: long entry pays spread."""

from src.bt.execution import execute_signal, create_execution_params
from src.bt.state import TradeSignal, Candle, ActionType
from src.utils import get_ts


def test_long_entry_pays_spread():
    params = create_execution_params(
        spread_bps=10.0, slippage_bps=2.0, fixed_commission=0.5
    )
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    candle = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.5,
        volume=1000,
    )
    fill = execute_signal(signal, candle, params)
    assert fill.executed_price > candle.open  # pays spread
    assert fill.commission == 0.5
