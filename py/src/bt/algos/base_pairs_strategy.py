from typing import List, Optional
import pandas as pd
from collections import defaultdict

from src.bt.types import TradeSignal, ActionType, TradeExitReason


class BasePairsStrategy:
    """
    Base class for pair trading strategies.
    Provides common utilities for generating trading signals.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.pending_ticks = defaultdict(dict)

    def _long(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        z_score: float,
        hedge_beta: Optional[float] = None,
    ) -> TradeSignal:
        """Create a long signal."""
        return TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=price,
            hedge_beta=hedge_beta,
        )

    def _short(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        z_score: float,
        hedge_beta: Optional[float] = None,
    ) -> TradeSignal:
        """Create a short signal."""
        return TradeSignal(
            action=ActionType.short,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=price,
            hedge_beta=hedge_beta,
        )

    def _close(
        self, symbol: str, timestamp: pd.Timestamp, price: float, z_score: float
    ) -> TradeSignal:
        """Create a close signal."""
        return TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            z_score=z_score,
            timestamp=timestamp,
            price=price,
            reason=TradeExitReason.regression,
        )
