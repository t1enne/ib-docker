from typing import List, Dict
import pandas as pd
from collections import defaultdict

from src.bt.types import TradeSignal, ActionType
from src.utils import validate_schema

props_schema = {
    "entry_threshold": float,
    "rolling_window_size": int,
}


class BasePairsStrategy:
    """
    Abstract base class for pair trading strategies.

    Provides common interface and utilities for processing tick data,
    managing positions, and generating trading signals.
    """

    def __init__(self, symbols: List[str], hdata: dict[str, pd.DataFrame], **kwargs):
        if not validate_schema(kwargs, props_schema):
            raise ValueError("wrong parameters")

        self.symbols = symbols
        self.entry_threshold: float = kwargs.get("entry_threshold", 2.0)
        self.rolling_window_size: int = kwargs.get("rolling_window_size", 100)
        # Common buffers
        self.z_scores: Dict[pd.Timestamp, float] = dict()
        self.pending_ticks = defaultdict(dict)
        self.hdata = hdata

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
