"""Factory functions for creating initial states."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

from src.bt.state.types import (
    PortfolioState,
    MarketDataState,
    ModelState,
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


def create_empty_market_data(symbols: List[str]) -> MarketDataState:
    """Create empty market data state."""
    return MarketDataState(timestamps=(), bars=(), symbols=tuple(symbols))


def create_initial_model_state(
    symbols: List[str], rolling_window_size: int
) -> ModelState:
    """Create initial model state."""
    return ModelState(
        z_score=0.0,
        current_regime=None,
        price_buffers=(),
        market_data=create_empty_market_data(symbols),
        hedge_beta=1.0,
    )


def create_initial_backtest_state(
    symbols: List[str],
    initial_capital: float,
    start_timestamp: pd.Timestamp,
    rolling_window_size: int,
) -> BacktestState:
    """Create initial backtest state."""
    return BacktestState(
        portfolio=create_initial_portfolio(initial_capital, start_timestamp),
        timestamp=None,
        pending_signals=[],
        model_state=create_initial_model_state(symbols, rolling_window_size),
        risk_events=(),
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
