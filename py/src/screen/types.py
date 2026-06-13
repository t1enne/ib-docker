"""Type definitions for the screen module.

All domain types are immutable (frozen dataclasses) — following
the same FP principles as the rest of the codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd


# ── Data types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenResult:
    """Result of screening a single symbol.

    Immutable — a snapshot of what the screen computed for one ticker.
    """

    symbol: str
    signal: Literal["long", "short", "neutral"]
    score: float  # Quantitative score for ranking (higher = more attractive)
    price: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ScreenResult") -> bool:
        """Default sort: descending by score."""
        return self.score > other.score


@dataclass(frozen=True)
class ScreenOutput:
    """Complete output of a screen run across a universe."""

    screen_name: str
    results: tuple[ScreenResult, ...]
    params: dict[str, Any] = field(default_factory=dict)


# ── Screen module Protocol ─────────────────────────────────────


class ScreenFn(Protocol):
    """Protocol for screen module instances.

    Each screen module exports a `make(symbols, params) -> ScreenFn` factory
    (following the same pattern as `StrategyProtocol` in src/bt/types.py).

    A ScreenFn instance carries per-run configuration and exposes:

        compute(symbol, candles) -> ScreenResult
        rank(results) -> list[ScreenResult]

    The factory receives the full symbol list and params dict at construction
    time so implementors can precompute universe-level statistics if needed.
    """

    def compute(
        self,
        symbol: str,
        candles: pd.DataFrame,
    ) -> ScreenResult:
        """Score/classify a single symbol given its candle history.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            candles: OHLCV DataFrame with DatetimeIndex and columns
                     open, high, low, close, volume

        Returns:
            ScreenResult with score, signal direction, and metadata.
        """
        ...

    def rank(
        self,
        results: list[ScreenResult],
    ) -> list[ScreenResult]:
        """Sort screen results by score descending.

        Override to implement custom ranking logic
        (e.g., penalize low volume, enforce minimum price).
        """
        ...
