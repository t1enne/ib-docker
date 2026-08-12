"""Factory functions for creating initial states."""

from typing import List, Optional

import pandas as pd

from src.bt.engine.candle_store import CandleStore
from src.bt.state.types import (
    PortfolioState,
    BacktestState,
    EquityPoint,
    ExecutionParams,
    RiskConfig,
)


def create_initial_portfolio(
    initial_capital: float, start_timestamp: pd.Timestamp
) -> PortfolioState:
    """Create initial empty portfolio state."""
    return PortfolioState(
        cash=initial_capital,
        positions={},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=start_timestamp,
                equity=initial_capital,
                cash=initial_capital,
                positions_value=0.0,
            ),
        ),
        initial_capital=initial_capital,
    )


def create_initial_backtest_state(
    symbols: List[str],
    initial_capital: float,
    start_timestamp: pd.Timestamp,
    rolling_window_size: Optional[int] = None,
) -> BacktestState:
    """Create initial backtest state."""
    return BacktestState(
        portfolio=create_initial_portfolio(initial_capital, start_timestamp),
        timestamp=None,
        pending_signals={},
        risk_events=(),
        candles=CandleStore({}),
    )


def create_execution_params(
    spread_bps: float = 5.0, slippage_bps: float = 2.0, fixed_commission: float = 0.5
) -> ExecutionParams:
    """Create execution parameters."""
    return ExecutionParams(
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        fixed_commission=fixed_commission,
    )


def create_risk_config(
    stop_loss_pct: float, take_profit_pct: float, trailing_stop: bool = False
) -> RiskConfig:
    """Create risk configuration."""
    return RiskConfig(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop=trailing_stop,
    )
