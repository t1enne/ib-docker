from src.bt.algos.z_model import TrainedZModel
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
    last_price: float
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
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    stability: float = 0.0
    omega_ratio: float = 0.0


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
class ExecutionParams:
    spread_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass
class FillEvent:
    signal: "TradeSignal"
    filled_qty: float
    executed_price: float
    commission: float
    slippage: float


@dataclass
class TradeSignal:
    action: ActionType
    symbol: str
    z_score: float
    timestamp: pd.Timestamp
    price: float
    reason: str = ""


class StrategyProtocol(Protocol):
    """Protocol for strategy classes that receive trained models."""

    def set_model(self, model: TrainedZModel) -> None: ...
    def on_tick(self, tick: Tick) -> List[TradeSignal]:
        """Process a tick and return new state and signals."""
        ...

    def get_z_scores(self) -> Optional[pd.DataFrame]:
        """Return DataFrame with 'z' column indexed by timestamp, or None if no z-scores."""
        ...
