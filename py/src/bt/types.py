from dataclasses import dataclass
import pandas as pd
from typing import List, Optional, Any, Protocol, Union, TypedDict, Dict
from enum import Enum


class ActionType(Enum):
    short = "short"
    long = "long"
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


@dataclass
class ZScoreState:
    scores: List[float]
    timestamps: List[pd.Timestamp]
    scores_synced: List[float]
    timestamps_synced: List[pd.Timestamp]


@dataclass
class RegimeState:
    labels: List[Optional[int]]
    probs: List[Optional[List[float]]]
    timestamps: List[pd.Timestamp]


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
    close_reason: Optional[TradeExitReason] = None


@dataclass
class StopLossEvent:
    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "sl"


@dataclass
class TakeProfitEvent:
    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "tp"


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
class BacktestResults:
    pf: PortfolioResult
    data: pd.DataFrame
    z_scores: pd.DataFrame
    regimes: Optional[pd.DataFrame]


@dataclass(frozen=True)
class EngineWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class Tick:
    timestamp: pd.Timestamp
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
    fixed_commission: float = 0.5


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
    reason: Optional[TradeExitReason] = TradeExitReason.none


@dataclass
class StrategyConfig:
    name: str
    strategy_type: str
    symbols: list[str]
    entry_z: float
    exit_z: float
    stop_loss: float
    take_profit: float
    initial_capital: float
    position_size: float
    commission: float
    training_start: str
    training_end: str
    trading_start: str
    trading_end: str
    rolling_window_size: int
    plot: bool
    bar: str
    hmm_floating_window: Optional[int] = None
    hmm_retrain_interval: Optional[int] = None


class StrategyType(Enum):
    PND = "pnd"
    SPREAD = "spread"


class StrategyProtocol(Protocol):
    """Protocol for strategy classes.

    Strategies receive a model object at construction that provides access to
    features and historical data. Use self.model.z_score, self.model.market_data,
    etc. from within your strategy.
    """

    model: Any  # StrategyModel instance

    def on_tick(self, tick: Tick, open_trade: Optional[Trade]) -> List[TradeSignal]:
        """Process a tick and return trading signals.

        Access computed features via self.model:
            - self.model.z_score          # Current z-score
            - self.model.current_regime   # Current HMM regime
            - self.model.market_data      # Historical OHLCV

        Apply indicators to market data:
            from src.bt.indicators import ema
            ema_9 = ema(self.model.market_data[-14:].close, 9)
        """
        ...


RiskEvent = Union[StopLossEvent, TakeProfitEvent]
