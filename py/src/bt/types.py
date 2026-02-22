from dataclasses import dataclass
import pandas as pd
from typing import List, Optional, Any, Protocol, Union, TypedDict, Dict
from enum import Enum

from src.bt.state import (
    Tick,
    Trade,
    TradeSignal,
    FillEvent,
    StopLossEvent,
    TakeProfitEvent,
    ExecutionParams,
    PortfolioResult,
    BacktestResults,
    ActionType,
    TradeStatus,
    TradeExitReason,
)


class ZScoreState:
    scores: List[float]
    timestamps: List[pd.Timestamp]
    scores_synced: List[float]
    timestamps_synced: List[pd.Timestamp]


class RegimeState:
    labels: List[Optional[int]]
    probs: List[Optional[List[float]]]
    timestamps: List[pd.Timestamp]


@dataclass(frozen=True)
class EngineWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


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
