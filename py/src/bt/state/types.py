"""State dataclasses for functional backtesting engine.

All state is immutable (frozen dataclasses) to enable:
- Easy testing and debugging
- State snapshots and replay
- Time travel debugging
- Deterministic behavior
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any, List, Union, Literal, TYPE_CHECKING
from enum import Enum, auto
import pandas as pd
import numpy as np
from typing import FrozenSet

if TYPE_CHECKING:
    from src.bt.types import PlotConfig


class ActionType(Enum):
    long = "long"
    short = "short"
    close = "close"


class TradeStatus(Enum):
    open = "open"
    closed = "closed"
    stopped = "stopped"


class TradeExitReason(Enum):
    sl = "sl"
    tp = "tp"
    end = "end"
    regression = "regression"
    none = "none"
    signal = "signal"


@dataclass(frozen=True)
class Tick:
    """A single OHLCV tick."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Position:
    """An open position."""

    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float]
    take_profit: Optional[float]
    last_price: float


@dataclass
class Trade:
    """A completed trade record."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    last_price: float
    z_score: Optional[float]
    symbol: str
    position: ActionType
    qty: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    status: TradeStatus = TradeStatus.open
    close_reason: Optional[Any] = None


@dataclass(frozen=True)
class EquityPoint:
    """Single point in equity curve."""

    timestamp: pd.Timestamp
    equity: float
    cash: float
    positions_value: float


@dataclass(frozen=True)
class PortfolioState:
    """Immutable portfolio state."""

    cash: float
    positions: Dict[str, Position]  # Symbol -> Position
    trades: Tuple[Trade, ...]
    equity_curve: Tuple[EquityPoint, ...]
    initial_capital: float


@dataclass(frozen=True)
class TradeSignal:
    """Signal to enter/exit a position."""

    action: ActionType
    symbol: str
    timestamp: pd.Timestamp
    price: float
    qty: float = 0.0
    reason: Optional[Any] = None
    z_score: Optional[float] = None
    hedge_beta: Optional[float] = None


@dataclass(frozen=True)
class FillEvent:
    """Result of executing a signal."""

    signal: TradeSignal
    filled_qty: float
    executed_price: float
    commission: float
    slippage: float
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class StopLossEvent:
    """Risk event for stop loss."""

    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "sl"


@dataclass(frozen=True)
class TakeProfitEvent:
    """Risk event for take profit."""

    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "tp"


RiskEvent = Tuple[StopLossEvent, TakeProfitEvent]


@dataclass(frozen=True)
class ExecutionParams:
    """Parameters for execution."""

    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    fixed_commission: float = 0.5


@dataclass(frozen=True)
class RiskConfig:
    """Parameters for risk management."""

    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: bool = False


@dataclass(frozen=True)
class MarketDataState:
    """Immutable market data history."""

    symbols: Tuple[str, ...]


@dataclass(frozen=True)
class ModelState:
    """Strategy model computations."""

    z_score: Optional[float]
    current_regime: Optional[int]
    price_buffers: Tuple[Dict[str, float], ...]
    market_data: MarketDataState
    hedge_beta: float = 1.0
    correlation_model: Optional[Any] = None
    resample_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    resample_anchor: Dict[str, pd.Timestamp] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestState:
    """Complete state snapshot at any point in time."""

    portfolio: PortfolioState
    timestamp: Optional[pd.Timestamp]
    pending_signals: List[TradeSignal]
    model_state: ModelState
    risk_events: Tuple[Any, ...]  # RiskEvent tuple
    candles: pd.DataFrame


@dataclass(frozen=True)
class PortfolioResult:
    """Final portfolio results."""

    total_return: float
    sharpe_ratio: float
    trades: Tuple[Trade, ...]
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


@dataclass(frozen=True)
class BacktestResults:
    """Complete backtest results."""

    pf: PortfolioResult
    data: pd.DataFrame
    z_scores: pd.DataFrame
    regimes: Optional[pd.DataFrame]
    final_state: BacktestState
    plot_config: Optional["PlotConfig"] = None
