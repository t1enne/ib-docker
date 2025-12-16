from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd
from src.bt.types import TradeSignal, ActionType


@dataclass
class StrategyConfig:
    entry_threshold: float
    exit_threshold: float
    rolling_window_size: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        if (
            self.entry_threshold <= 0
            or self.exit_threshold <= 0
            or self.rolling_window_size <= 0
        ):
            raise ValueError("Thresholds and window size must be positive")


@dataclass
class StrategyState:
    symbols: List[str]
    config: StrategyConfig
    historical_data: Dict[str, pd.DataFrame] = field(
        default_factory=dict
    )  # Mutable for performance
    positions: Dict[str, float] = field(default_factory=dict)
    z_scores: Dict[pd.Timestamp, float] = field(default_factory=dict)
    pending_ticks: Dict[pd.Timestamp, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.positions:
            self.positions = {symbol: 0.0 for symbol in self.symbols}
        if not self.historical_data:
            self.historical_data = {
                symbol: pd.DataFrame({"timestamp": [], "close": []})
                for symbol in self.symbols
            }

    def copy_with_updates(self, **updates) -> "StrategyState":
        # Create a new state with updates, keeping historical_data reference for perf
        return StrategyState(
            symbols=self.symbols,
            config=self.config,
            historical_data=updates.get("historical_data", self.historical_data),
            positions=updates.get("positions", self.positions).copy(),
            z_scores=updates.get("z_scores", self.z_scores).copy(),
            pending_ticks=updates.get("pending_ticks", self.pending_ticks).copy(),
        )


def init_state(symbols: List[str], **config_kwargs) -> StrategyState:
    config = StrategyConfig(**config_kwargs)
    return StrategyState(symbols=symbols, config=config)


def add_historical_data(
    state: StrategyState, data: Dict[str, pd.DataFrame]
) -> StrategyState:
    # For performance, update in-place but return new state
    new_historical = state.historical_data.copy()
    for symbol, df in data.items():
        new_historical[symbol] = pd.DataFrame(
            {"timestamp": df.index, "close": df["Close"]}
        )
    return state.copy_with_updates(historical_data=new_historical)


def get_z_score(state: StrategyState, ts: pd.Timestamp) -> float:
    return state.z_scores.get(ts, 0.0)


def create_signal(
    action: ActionType, symbol: str, ts: pd.Timestamp, state: StrategyState
) -> TradeSignal:
    return TradeSignal(
        action=action,
        symbol=symbol,
        z_score=get_z_score(state, ts),
        timestamp=ts,
        price=state.pending_ticks[ts][symbol],
    )

