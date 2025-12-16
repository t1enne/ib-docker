from abc import ABC, abstractmethod
from typing import List, Dict
import asyncio
import pandas as pd

from src.bt.types import TradeSignal, ActionType


class BasePairsStrategy(ABC):
    """
    Abstract base class for pair trading strategies.

    Provides common interface and utilities for processing tick data,
    managing positions, and generating trading signals.
    """

    def __init__(self, symbols: List[str], **kwargs):
        self.symbols = symbols
        self.entry_threshold = kwargs.get("entry_threshold", 2.0)
        self.exit_threshold = kwargs.get("exit_threshold", 0.5)

        # Common buffers
        self.historical_data: Dict[str, List] = {symbol: [] for symbol in symbols}
        self.pending_ticks: Dict[pd.Timestamp, Dict[str, float]] = dict()

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

    def _long(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        """Create a long signal."""
        return TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _short(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        """Create a short signal."""
        return TradeSignal(
            action=ActionType.short,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )

    def _close(
        self, symbol: str, z_score: float, timestamp: pd.Timestamp
    ) -> TradeSignal:
        """Create a close signal."""
        return TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=self.pending_ticks[timestamp][symbol],
        )
