"""State module for functional backtesting."""

from src.bt.state.types import (
    Candle,
    Position,
    Trade,
    EquityPoint,
    PortfolioState,
    TradeSignal,
    FillEvent,
    StopLossEvent,
    TakeProfitEvent,
    RiskEvent,
    ExecutionParams,
    RiskConfig,
    MarketDataState,
    ModelState,
    BacktestState,
    PortfolioResult,
    BacktestResults,
    ActionType,
    TradeStatus,
    TradeExitReason,
)

from src.bt.state.factories import (
    create_initial_portfolio,
    create_empty_market_data,
    create_initial_model_state,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
)

__all__ = [
    # Types
    "Candle",
    "Position",
    "Trade",
    "EquityPoint",
    "PortfolioState",
    "TradeSignal",
    "FillEvent",
    "StopLossEvent",
    "TakeProfitEvent",
    "RiskEvent",
    "ExecutionParams",
    "RiskConfig",
    "MarketDataState",
    "ModelState",
    "BacktestState",
    "PortfolioResult",
    "BacktestResults",
    "ActionType",
    "TradeStatus",
    "TradeExitReason",
    # Factories
    "create_initial_portfolio",
    "create_empty_market_data",
    "create_initial_model_state",
    "create_initial_backtest_state",
    "create_execution_params",
    "create_risk_config",
]
