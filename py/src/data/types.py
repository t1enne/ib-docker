"""Type definitions for the sync module.

All domain types are immutable — following FP principles:
data over control flow, make invalid states unrepresentable.

Types defined here:
  CandleDict     — TypedDict for OHLCV candle data (structural product type)
  UniverseConf   — frozen config loaded from universe .json
  SyncResult     — frozen result of a sync operation
  FetchPlan      — frozen per-symbol gap plan for dry-run
  PreviewResult  — frozen result of a dry-run preview
  ProgressFn     — Protocol for progress callbacks (I/O boundary)
  ISymbol        — dataclass mirroring SymbolSchema peewee model
  ICandle        — dataclass mirroring CandleSchema peewee model
  SymbolSchema   — peewee ORM model for symbols
  CandleSchema   — peewee ORM model for candles
  db             — SqliteDatabase instance (shared by models)
"""

import atexit
import os

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Protocol, TypedDict

from peewee import Model, IntegerField, CharField, FloatField, SqliteDatabase


# ── Database instance ───────────────────────────────────────────

db_path = os.path.join(os.getcwd(), "..", "data", "db.sqlite")
db = SqliteDatabase(db_path, pragmas={"journal_mode": "wal"})
atexit.register(db.close)


# ── Dataclass mirrors of ORM models ────────────────────────────


@dataclass
class ISymbol:
    """Dataclass matching SymbolSchema Peewee model."""

    conid: int
    ticker: str
    market: str
    currency: str
    name: Optional[str] = None


@dataclass
class ICandle:
    """Dataclass matching CandleSchema Peewee model."""

    conid: int
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Peewee ORM models ──────────────────────────────────────────


class SymbolSchema(Model):
    conid = IntegerField(primary_key=True)
    ticker = CharField()
    name = CharField(null=True)
    market = CharField()
    currency = CharField()

    class Meta:
        database = db
        table_name = "symbol"


class CandleSchema(Model):
    conid = IntegerField()
    ticker = CharField()
    timestamp = IntegerField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    class Meta:
        database = db
        table_name = "candle"


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
    """Configuration loaded from universe .json.

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
