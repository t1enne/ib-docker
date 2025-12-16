from dataclasses import dataclass
import pandas as pd
from typing import List, Optional, Any, Protocol
from enum import Enum


class ActionType(Enum):
    short = "short"
    long = "long"
    close = "close"


class TradeStatus(Enum):
    open = "open"
    closed = "closed"
    stopped = "stopped"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    z_score: float
    symbol: str
    position: ActionType
    qty: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    status: TradeStatus = TradeStatus.open
    close_reason: Optional[str] = None


@dataclass
class PortfolioResult:
    total_return: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    trades: List[Trade]
    equity_curve: pd.Series


@dataclass
class Tick:
    timestamp: Any
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeSignal:
    action: ActionType
    symbol: str
    z_score: float
    timestamp: pd.Timestamp
    price: float


class StrategyProtocol(Protocol):
    """Protocol for strategy classes."""

    def on_tick(self, tick: Tick) -> List[TradeSignal]:
        """Process a tick and return new state and signals."""
        ...
