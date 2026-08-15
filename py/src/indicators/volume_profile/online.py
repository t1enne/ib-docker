"""Incremental (online) Volume Profile accumulator for the backtest hot path.

Where the batch :func:`src.indicators.volume_profile.pure.volume_profile`
re-profiles a whole candle window on every call (O(window × bins)), this class
maintains the same histogram *incrementally*: each ``observe`` add a candle
and (for a rolling window) evicts the oldest in O(num_bins), so the common
case is near-free and the exact same bins as the batch function are kept.

The sample exactness trick
--------------------------
The overlap-weighted bin math matches ``volume_profile`` exactly. When a new
candle is inside the current price frame the accumulator just adds its overlap
vector and subtracts the evicted candle's — O(num_bins), no rebin. Only when
the window's price range *changes* (a new high/low, or the evicted candle was
an extreme so the range shrinks) does it rebuild the entire window in the new
frame once — rare, and the only O(window × bins) step.

Two accumulation modes:
- ``window=N`` : rolling profile over the last N candles (recommended for live
  strats — decays old context automatically).
- ``window=None``: accumulate from inception; profile only grows. Matches an
  anchored / from-first-candle profile.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.indicators.volume_profile.pure import (
    _aggregate_overlap,
    _candle_overlap,
    _levels_to_profile,
)
from src.indicators.volume_profile.types import VolumeProfileSnapshot


class OnlineVolumeProfile:
    """Accumulating Volume Profile, owned by a strategy via ``ctx.shared``.

    Feed one candle per bar with :meth:`observe`; it returns derived levels
    (POC / Value Area High/Low) that become ``ready`` after ``warmup_bars``
    candles have accumulated. State lives entirely on the instance, so one
    per symbol (or keyed in ``ctx.shared``) is safe across split/sweep workers
    when reconstructed fresh per run.

    Parameters
    ----------
    num_bins:
        Number of equal-width price bins (default 50).
    value_area_pct:
        Fraction of total volume that defines the Value Area (default 0.70).
    window:
        Rolling window length in candles. ``None`` accumulates without eviction.
    warmup_bars:
        Minimum candles before ``ready`` becomes True (default 50).
    """

    def __init__(
        self,
        num_bins: int = 50,
        value_area_pct: float = 0.70,
        window: int | None = 200,
        warmup_bars: int = 50,
    ) -> None:
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        if not 0.0 < value_area_pct <= 1.0:
            raise ValueError(f"value_area_pct must be in (0, 1], got {value_area_pct}")
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 or None, got {window}")

        self._num_bins = num_bins
        self._value_area_pct = value_area_pct
        self._window = window
        self._warmup_bars = warmup_bars

        # Rolling candle buffer: (low, high, volume).
        self._candles: deque[tuple[float, float, float]] = deque()
        self._n = 0  # total candles observed (incl. evicted)

        # Current bin frame + histogram (None until first valid candle).
        self._lo: float | None = None
        self._hi: float | None = None
        self._bin_lo: np.ndarray | None = None
        self._bin_hi: np.ndarray | None = None
        self._mids: np.ndarray | None = None
        self._bin_volume: np.ndarray | None = None
        self._total: float = 0.0

    # ------------------------------------------------------------------
    # read-only state
    # ------------------------------------------------------------------

    @property
    def n_observed(self) -> int:
        """Total candles passed to ``observe`` (including evicted ones)."""
        return self._n

    @property
    def n_candles(self) -> int:
        """Candles currently in the profile (the active buffer size)."""
        return len(self._candles)

    @property
    def ready(self) -> bool:
        """True once enough candles have accumulated to derive levels."""
        return self._bin_volume is not None and len(self._candles) >= self._warmup_bars

    @property
    def bins(self) -> np.ndarray:
        """Midpoints of the current price bins (empty before the first candle)."""
        if self._mids is None:
            return np.array([], dtype=float)
        return self._mids

    @property
    def range(self) -> tuple[float, float] | None:
        """Current price range ``(low, high)`` of the active window, if any."""
        if self._lo is None or self._hi is None:
            return None
        return (self._lo, self._hi)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def snapshot(self) -> VolumeProfileSnapshot:
        """Return the current derived levels without feeding a candle.

        Call when a symbol has no new bar yet (e.g. its candle series is empty)
        so the strategy still sees the latest profile state.
        """
        return self._snapshot()

    def reset(self) -> None:
        """Clear all accumulated candles and histograms.

        Reconstructing the instance per run also achieves this; ``reset()`` is
        provided for the runner's defensive ``reset_global()`` protocol.
        """
        self._candles.clear()
        self._n = 0
        self._lo = None
        self._hi = None
        self._bin_lo = None
        self._bin_hi = None
        self._mids = None
        self._bin_volume = None
        self._total = 0.0

    def observe(
        self,
        low: float,
        high: float,
        volume: float,
    ) -> VolumeProfileSnapshot:
        """Feed one candle and return the derived profile snapshot.

        Invalid rows (NaN / ``high < low`` / non-positive volume) are ignored
        for the profile but still counted toward ``n_observed`` so callers can
        tell the candle was seen.

        Returns a snapshot with ``ready=False`` until ``warmup_bars`` valid
        candles have accumulated.
        """
        if low != low or high != high or volume != volume:  # NaN guard
            self._n += 1
            return self._snapshot()
        if volume <= 0.0 or high < low:
            self._n += 1
            return self._snapshot()

        self._n += 1
        self._candles.append((low, high, volume))
        evicted: tuple[float, float, float] | None = None
        if self._window is not None and len(self._candles) > self._window:
            evicted = self._candles.popleft()

        new_lo = min(c[0] for c in self._candles)
        new_hi = max(c[1] for c in self._candles)

        # Rebuild if the range changed (expanded by the new candle or shrunk by
        # evicting an extreme) or this is the first candle. Otherwise the frame
        # is unchanged and we can add/evict in O(num_bins).
        frame_ready = (
            self._bin_volume is not None
            and self._lo is not None
            and self._hi is not None
        )
        range_changed = frame_ready and (
            new_lo < self._lo
            or new_hi > self._hi
            or new_lo > self._lo
            or new_hi < self._hi
        )
        if not frame_ready or range_changed:
            self._rebuild(new_lo, new_hi)
        else:
            # Exact incremental update inside the stable frame.
            assert self._bin_lo is not None and self._bin_hi is not None
            assert self._bin_volume is not None
            added = _candle_overlap(low, high, self._bin_lo, self._bin_hi) * volume
            self._bin_volume += added
            self._total += volume
            if evicted is not None:
                removed = (
                    _candle_overlap(evicted[0], evicted[1], self._bin_lo, self._bin_hi)
                    * evicted[2]
                )
                self._bin_volume -= removed
                self._total -= evicted[2]

        return self._snapshot()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _rebuild(self, new_lo: float, new_hi: float) -> None:
        """Re-derive the bin frame for ``[new_lo, new_hi]`` and histogram by
        aggregating every active candle with the batch overlap math (exact)."""
        self._lo = new_lo
        self._hi = new_hi
        if not (new_hi > new_lo):
            # Every candle is a single flat price level -> one degenerate bin.
            edges = np.linspace(new_lo, new_lo + 1e-9, self._num_bins + 1)
        else:
            edges = np.linspace(new_lo, new_hi, self._num_bins + 1)
        self._bin_lo = edges[:-1]
        self._bin_hi = edges[1:]
        self._mids = (self._bin_lo + self._bin_hi) / 2.0

        if not self._candles:
            self._bin_volume = np.zeros(self._num_bins, dtype=float)
            self._total = 0.0
            return

        lows = np.array([c[0] for c in self._candles], dtype=float)
        highs = np.array([c[1] for c in self._candles], dtype=float)
        vols = np.array([c[2] for c in self._candles], dtype=float)
        self._bin_volume = _aggregate_overlap(
            lows, highs, vols, self._bin_lo, self._bin_hi
        )
        self._total = float(vols.sum())

    def _snapshot(self) -> VolumeProfileSnapshot:
        ready = self._bin_volume is not None and len(self._candles) >= self._warmup_bars
        assert self._bin_lo is not None and self._bin_hi is not None
        assert self._mids is not None and self._bin_volume is not None

        if not ready:
            return VolumeProfileSnapshot(
                bin_price=self._mids,
                volume=self._bin_volume.copy(),
                total_volume=self._total,
                poc=None,
                poc_volume=0.0,
                vah=None,
                val=None,
                value_volume=0.0,
                n_candles=len(self._candles),
                ready=False,
            )

        prof = _levels_to_profile(
            self._bin_lo,
            self._bin_hi,
            self._mids,
            self._bin_volume,
            self._value_area_pct,
        )
        return VolumeProfileSnapshot(
            bin_price=prof.bin_price,
            volume=prof.volume,
            total_volume=prof.total_volume,
            poc=prof.poc,
            poc_volume=prof.poc_volume,
            vah=prof.vah,
            val=prof.val,
            value_volume=prof.value_volume,
            n_candles=len(self._candles),
            ready=True,
        )
