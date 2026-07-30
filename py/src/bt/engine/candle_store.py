"""CandleStore — lazy DataFrame view over incremental numpy column arrays.

Single source of truth for OHLCV data during backtest. Wraps the mutable
``CandleRows`` accumulator by reference. Two access paths:

* ``.latest(sym, iv)`` / ``.count(sym, iv)`` — O(1) numpy reads, zero allocation
* ``store[(sym, iv)]`` / ``store.get(...)`` / iteration — lazy DataFrame build
  (``Mapping[(str, str), DataFrame]`` for drop-in strategy compatibility)

See ``src/bt/engine/backtest.py`` for the ``CandleRows`` layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Timestamp

# ---------------------------------------------------------------------------
# shared type
# ---------------------------------------------------------------------------

# Per-candle row accumulator — column-major numpy arrays keyed by (symbol, interval).
# Each value: {"timestamp": ndarray, "open": ndarray, …, "_len": array([N])}.
CandleRows = dict[tuple[str, str], dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# CandleStore
# ---------------------------------------------------------------------------


class CandleStore(Mapping[tuple[str, str], "DataFrame"]):
    """Lazy DataFrame view over incremental numpy column arrays.

    Wraps a ``CandleRows`` dict by reference — rows are appended in-place
    by the backtest loop.  Strategies read through this store.

    *Cursor* — a timestamp ceiling set via ``.advance(ts)`` by the engine
    before each strategy invocation.  DataFrames built via the Mapping
    interface are truncated to the cursor, so a strategy never sees data
    from future timestamps.  ``.latest()`` and ``.count()`` are not
    cursor-truncated (they return the absolute latest value known to the
    accumulator).

    ``Mapping`` interface: ``__getitem__``, ``get``, ``__contains__``,
    ``__len__``, ``__iter__``, ``keys``, ``items``, ``values``.
    """

    __slots__ = ("_rows", "_cursor")

    def __init__(
        self,
        rows: CandleRows,
        cursor: Timestamp | None = None,
    ) -> None:
        self._rows: CandleRows = rows
        self._cursor: Timestamp | None = cursor

    # -- mutation (called by engine, not strategies) --------------------

    def advance(self, ts: Timestamp) -> None:
        self._cursor = ts

    # -- fast path: O(1) from numpy, no DataFrame build -----------------

    def latest(self, sym: str, interval: str) -> float | None:
        """Return the most recent close for *sym* at *interval*, or None."""
        cols = self._rows.get((sym, interval))
        if cols is None:
            return None
        n = int(cols["_len"][0])
        if n == 0:
            return None
        return float(cols["close"][n - 1])

    def count(self, sym: str, interval: str) -> int:
        """Return the number of accumulated bars for *sym* at *interval*."""
        cols = self._rows.get((sym, interval))
        if cols is None:
            return 0
        return int(cols["_len"][0])

    # -- Mapping interface (full DataFrame, built on demand) ------------

    def _build_df(self, key: tuple[str, str]) -> DataFrame:
        cols = self._rows[key]
        n = int(cols["_len"][0])
        ts_arr = cols["timestamp"][:n]

        # Truncate to cursor
        if self._cursor is not None:
            cursor_ns = np.datetime64(self._cursor.to_datetime64())
            idx = np.searchsorted(ts_arr, cursor_ns, side="right")
            n = int(idx)
            ts_arr = ts_arr[:n]
            if n == 0:
                return pd.DataFrame(
                    {
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                    },
                    index=pd.DatetimeIndex([]),
                )

        return pd.DataFrame(
            {
                "open": cols["open"][:n],
                "high": cols["high"][:n],
                "low": cols["low"][:n],
                "close": cols["close"][:n],
                "volume": cols["volume"][:n],
            },
            index=pd.DatetimeIndex(ts_arr),
        )

    def __getitem__(self, key: tuple[str, str]) -> DataFrame:
        if key not in self._rows:
            raise KeyError(key)
        return self._build_df(key)

    def get(self, key: object, default: object = None) -> DataFrame | None:  # ty: ignore[invalid-method-override]
        if not isinstance(key, tuple) or key not in self._rows:
            return cast("DataFrame | None", default)
        return self._build_df(cast("tuple[str, str]", key))

    def __contains__(self, key: object) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def keys(self):
        return self._rows.keys()

    def items(self):
        for k in self._rows:
            yield k, self._build_df(k)

    def values(self):
        for k in self._rows:
            yield self._build_df(k)
