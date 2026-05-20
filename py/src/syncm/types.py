"""Type definitions for the sync module.

All domain types are immutable — following FP principles:
data over control flow, make invalid states unrepresentable.

Types defined here:
  CandleDict     — TypedDict for OHLCV candle data (structural product type)
  UniverseConf   — frozen config loaded from universe.yml
  SyncResult     — frozen result of a sync operation
  FetchPlan      — frozen per-symbol gap plan for dry-run
  PreviewResult  — frozen result of a dry-run preview
  ProgressFn     — Protocol for progress callbacks (I/O boundary)
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol, TypedDict


# ── Candle data ──────────────────────────────────────────────────


class CandleDict(TypedDict, total=True):
    """A single OHLCV candle from IBKR, ready for DB insertion.

    total=True means every key is required — invalid states
    (missing fields) are structurally impossible.
    """

    conid: int
    ticker: str
    timestamp: int  # milliseconds since epoch
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Configuration ────────────────────────────────────────────────


@dataclass(frozen=True)
class UniverseConf:
    """Configuration loaded from universe.yml.

    Frozen — configuration is immutable after loading.
    """

    symbols: list[str]
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    bar: str = "1h"


# ── Results ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncResult:
    """Result of a sync_data operation.

    Immutable — a snapshot of what happened during the sync.
    """

    resolved: int  # How many tickers resolved successfully
    fetched: list[int]  # Conids that had new data fetched
    gaps_found: int  # Total gap segments filled

    @property
    def total_fetched(self) -> int:
        """Number of symbols that received new data."""
        return len(self.fetched)


@dataclass(frozen=True)
class FetchPlan:
    """A plan for what gaps need to be fetched for a single symbol."""

    ticker: str
    conid: int
    gaps: list[tuple[datetime, datetime]]


@dataclass(frozen=True)
class PreviewResult:
    """Result of a dry-run preview — shows what would be fetched.

    Immutable — pure description, no side effects performed.
    """

    resolved: int  # How many tickers resolved
    total_gaps: int  # Total number of gap segments across all symbols
    plans: list[FetchPlan]  # Per-symbol gap plans


# ── Callbacks (I/O boundary) ─────────────────────────────────────


class ProgressFn(Protocol):
    """Callback for sync progress updates.

    Executed at the I/O boundary — the core logic remains pure.
    """

    def __call__(self, status: str, current: int, total: int) -> None: ...
