from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import pandas as pd

from src.bt.state import (  # noqa: F401
    Candle,
    TradeSignal,
    FillEvent,
    StopLossEvent,
    TakeProfitEvent,
    ExecutionParams,
    BacktestState,
    PortfolioState,
    PortfolioResult,
    BacktestResults,
    ActionType,
    TradeStatus,
    TradeExitReason,
    RiskConfig,
)
from src.bt.state import RiskEvent as StateRiskEvent


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
    stop_loss: float
    take_profit: float
    initial_capital: float
    position_size: float
    commission: float
    training_start: str
    training_end: str
    trading_start: str
    trading_end: str
    bars: list[str]
    # the strategy_params will be passed to the strategy raw
    strategy_params: dict
    model_params: dict = field(default_factory=dict)
    model_updater: dict | bool = False
    # the rolling_window_size will be used to retrain the models
    rolling_window_size: Optional[int] = None
    hmm_floating_window: Optional[int] = None
    hmm_retrain_interval: Optional[int] = None
    benchmark_symbols: list[str] = field(default_factory=lambda: ["SPY"])


RiskEvent = Union[StopLossEvent, TakeProfitEvent]


class StrategyFn(Protocol):
    """Protocol for strategy signal generation function."""

    def __call__(
        self,
        state: BacktestState,
        tick: Candle,
        params: Dict[str, Any],
    ) -> List[TradeSignal]: ...


class ModelUpdaterFn(Protocol):
    """Protocol for model state update function."""

    def __call__(
        self,
        state: BacktestState,
        tick: Candle,
    ) -> BacktestState: ...


class ExecutionFn(Protocol):
    """Protocol for signal execution function."""

    def __call__(
        self,
        signal: TradeSignal,
        tick: Candle,
        exec_params: ExecutionParams,
    ) -> FillEvent: ...


class PositionSizerFn(Protocol):
    """Protocol for position sizing and fill application."""

    def __call__(
        self,
        portfolio: PortfolioState,
        fill: FillEvent,
        sizing_params: Dict[str, float],
    ) -> PortfolioState: ...


class RiskCheckFn(Protocol):
    """Protocol for risk checking function."""

    def __call__(
        self,
        portfolio: PortfolioState,
        tick: Candle,
        risk_config: RiskConfig,
    ) -> Tuple[Tuple[StateRiskEvent, ...], PortfolioState]: ...


class ZScoreFn(Protocol):
    """Protocol for z-score calculation."""

    def __call__(
        self,
        price_buffers: Tuple[Dict[str, float], ...],
    ) -> Tuple[float, float]: ...


class DataLoaderFn(Protocol):
    """Protocol for data loading function."""

    def __call__(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        bar: str,
    ) -> pd.DataFrame: ...
