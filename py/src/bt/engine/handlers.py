"""Handler dataclasses for backtest execution and risk management.

These dataclasses hold the injectable functions for executing signals
and checking risk during backtesting.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from src.bt.state import (
    TradeSignal,
    Tick,
    FillEvent,
    ExecutionParams,
    PortfolioState,
    RiskConfig,
    RiskEvent,
    BacktestState,
)


@dataclass
class ExecutionHandler:
    """Handler for executing trade signals and applying fills.

    All fields are callable functions that can be swapped for testing.
    """

    execute_signal: Any
    execute_risk_event: Any
    apply_fill: Any


@dataclass
class RiskHandler:
    """Handler for checking and executing risk events."""

    check_risk: Any


def default_execution_handler() -> ExecutionHandler:
    """Create default execution handler with production functions."""
    from src.bt.execution.pure import execute_signal, execute_risk_event
    from src.bt.portfolio.pure import apply_fill

    return ExecutionHandler(
        execute_signal=execute_signal,
        execute_risk_event=execute_risk_event,
        apply_fill=apply_fill,
    )


def default_risk_handler() -> RiskHandler:
    """Create default risk handler with production functions."""
    from src.bt.risk.pure import check_risk

    return RiskHandler(
        check_risk=check_risk,
    )
