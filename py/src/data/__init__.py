"""Data module — IBKR market data download, query, and resampling.

Public API surface:
  data_group          — CLI group (`data dl/query/preview`)
  download            — fetch historical candles from IBKR into the DB
  preview             — dry-run gap analysis (what would be downloaded)
  load_universe_config— load/validate a universe .json file
  query_candles       — read candles back from the local DB
  get_connection      — raw sqlite connection
  resample_*          — OHLCV resampling utilities
  * types             — domain types (ISymbol, SyncResult, ...)

Legacy alias: ``sync_data`` is kept for backward compatibility with
existing callers, delegating to ``download``.
"""

from src.data.cli import data_group

from src.data.dl import download
from src.data.preview import preview
from src.data.symbols import load_universe_config, resolve_symbols

from src.data.types import (
    SymbolSchema,
    ISymbol,
    CandleSchema,
    CandleDict,
    FetchPlan,
    SyncResult,
    PreviewResult,
    UniverseConf,
    ProgressFn,
)
from src.data.db import query_candles, get_connection
from src.data.resample import resample_ohlcv, resample_multiindex

# Backward-compatible alias — the download path now lives in the `dl` module.
sync_data = download


__all__ = [
    "data_group",
    "download",
    "preview",
    "resolve_symbols",
    "load_universe_config",
    "sync_data",
    "SymbolSchema",
    "ISymbol",
    "CandleSchema",
    "CandleDict",
    "FetchPlan",
    "SyncResult",
    "PreviewResult",
    "UniverseConf",
    "ProgressFn",
    "query_candles",
    "get_connection",
    "resample_ohlcv",
    "resample_multiindex",
]
