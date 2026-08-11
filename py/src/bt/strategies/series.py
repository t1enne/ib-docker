"""SeriesView — a cursor-truncated numerical series for the strategy DSL.

Pine-flavoured replacement for raw ``pd.Series`` access inside strategies.
A ``SeriesView`` wraps a full numpy array (an indicator computed once over the
complete OHLCV feed) and a *lengther* that returns the number of bars that are
visible from the engine's current cursor.

The key safety property: **a SeriesView can never yield a future bar.** The
valid length is derived from the cursor shared with the ``CandleStore``, so
even ``view[-1]`` resolves to the *current* candle's value, never the next one.
This structurally prevents both lookahead and index misalignment that a hand
-rolled ``rolling()``/``shift()`` on ``state.candles`` tends to introduce.

Access cost is O(1): the lengther is a cheap numpy ``searchsorted`` (C-level,
logarithmic at worst), and slicing a numpy array is a view, not a copy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A callable that returns the number of bars currently visible at the cursor
# for the series this view is bound to.
Lengther = Callable[[], int]


class SeriesView:
    """Immutable cursor-truncated window over a full numeric series.

    ``values`` is the complete series (all bars in the feed for the symbol).
    ``lengther`` returns how many of those bars are visible *right now* (i.e.
    up to the engine's current timestamp). All indexing is relative to the
    cursor, so negative indices count back from the current bar and any
    request beyond the visible range is clamped to it.
    """

    __slots__ = ("_values", "_lengther")

    def __init__(self, values: np.ndarray, lengther: Lengther) -> None:
        # Keep the full array — never mutated after construction.
        self._values: np.ndarray = np.asarray(values, dtype=np.float64)
        self._lengther: Lengther = lengther

    # -- length & visibility -------------------------------------------------

    def _n(self) -> int:
        """Number of bars visible now (cursor-truncated, clamped to series)."""
        n = self._lengther()
        return max(0, min(n, len(self._values)))

    def __len__(self) -> int:
        return self._n()

    @property
    def visible(self) -> int:
        """Alias for the cursor-truncated length (readable, explicit)."""
        return self._n()

    # -- access --------------------------------------------------------------

    def __getitem__(self, i: int) -> float:
        """Return the bar at position ``i``.

        ``i`` is relative to the visible tail: ``-1`` is the current bar,
        ``-2`` the previous, etc. Absolute non-negative indices are clamped to
        the visible region (so ``view[0]`` is always the first visible bar).
        An empty view raises ``IndexError`` like an empty sequence.
        """
        n = self._n()
        if n <= 0:
            raise IndexError("SeriesView is empty (no visible bars)")
        if i < 0:
            idx = n + i
            if idx < 0:
                raise IndexError(f"index {i} out of range on {n}-bar view")
        else:
            idx = min(i, n - 1)
        return float(self._values[idx])

    def __iter__(self):
        n = self._n()
        for i in range(n):
            yield float(self._values[i])

    def __repr__(self) -> str:
        return f"SeriesView(visible={self._n()}, total={len(self._values)})"

    # -- numpy bridge (read-only, cursor-truncated) ---------------------------

    def to_array(self) -> np.ndarray:
        """Cursor-truncated numpy copy of the visible region (for testing/raw use)."""
        return self._values[: self._n()].copy()

    def last(self) -> float:
        """Current (cursor) bar value — equivalent to ``view[-1]``."""
        return self[-1]

    # -- Pine built-ins operating on a series (pure helpers) -----------------

    def nz(self, fallback: float = 0.0) -> float:
        """Current bar value, or ``fallback`` when NaN. (Pine ``nz``.)"""
        v = self[-1]
        return v if v == v else fallback

    def change(self, bars: int = 1) -> float:
        """Difference vs ``bars`` bars ago. (Pine ``change``.)"""
        cur = self[-1]
        if self._n() <= bars:
            return 0.0
        prev = self[-(bars + 1)]
        if cur != cur or prev != prev:
            return 0.0
        return cur - prev


__all__ = ["SeriesView", "Lengther"]
