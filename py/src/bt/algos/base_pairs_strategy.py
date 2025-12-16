from abc import ABC, abstractmethod
from typing import List, Dict
import asyncio
import pandas as pd
from collections import defaultdict

from src.bt.types import TradeSignal, ActionType
from src.utils import validate_schema

props_schema = {
    "entry_threshold": float,
    "exit_threshold": float,
    "rolling_window_size": int,
    "stop_loss": float,
    "take_profit": float,
}


class BasePairsStrategy(ABC):
    """
    Abstract base class for pair trading strategies.

    Provides common interface and utilities for processing tick data,
    managing positions, and generating trading signals.
    """

    def __init__(self, symbols: List[str], **kwargs):
        if not validate_schema(kwargs, props_schema):
            raise ValueError("wrong parameters")

        self.symbols = symbols
        self.entry_threshold = kwargs.get("entry_threshold")
        self.exit_threshold = kwargs.get("exit_threshold")
        self.rolling_window_size = kwargs.get("rolling_window_size")
        self.stop_loss = kwargs.get("stop_loss")
        self.take_profit = kwargs.get("take_profit")

        # Common buffers
        self.z_scores: Dict[pd.Timestamp, float] = dict()
        self.pending_ticks = defaultdict(dict)
        self.historical_data: Dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame() for symbol in symbols
        }
        # Position tracking: positive for long, negative for short
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}

    @abstractmethod
    async def process_data(
        self, ticks_queue: asyncio.Queue, order_queue: asyncio.Queue
    ):
        """
        Process incoming tick data and generate trading signals.

        Args:
            ticks_queue: Queue of incoming market ticks
            order_queue: Queue for outgoing trading signals
        """
        pass

    @abstractmethod
    def _calculate_signal(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """
        Calculate trading signals based on strategy logic.

        Args:
            timestamp: Current timestamp to process

        Returns:
            List of trading signals (can be empty)
        """
        pass

    def populate_historical_data(self, data: Dict[str, pd.DataFrame]):
        for symbol in data:
            df = data[symbol]
            self.historical_data[symbol] = pd.DataFrame(
                {"timestamp": df.index, "close": df["Close"]}
            )

    def _get_z(self, ts: pd.Timestamp) -> float:
        return self.z_scores[ts]

    def _long(self, symbol: str, timestamp: pd.Timestamp) -> TradeSignal:
        """Create a long signal."""
        return TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            z_score=self._get_z(timestamp),
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _short(self, symbol: str, timestamp: pd.Timestamp) -> TradeSignal:
        """Create a short signal."""
        return TradeSignal(
            action=ActionType.short,
            symbol=symbol,
            z_score=self._get_z(timestamp),
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _close(self, symbol: str, timestamp: pd.Timestamp) -> TradeSignal:
        """Create a close signal."""
        return TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=self._get_z(timestamp),
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )
