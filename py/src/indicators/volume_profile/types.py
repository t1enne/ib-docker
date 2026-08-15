"""Volume Profile — point-of-control / value-area types."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolumeProfile:
    """Fixed-range Volume Profile distribution over a set of candles.

    Aggregates trade volume across the price bins a range of candles traded
    within. The classic market-structure levels are derived from the resulting
    distribution:

    - ``poc``  Point of Control — midpoint of the bin with the highest volume
            (the dominant traded price level / magnet).
    - ``vah``  Value Area High — upper edge of the Value Area, the tight price
            band around the POC covering ``value_area_pct`` of total volume.
    - ``val``  Value Area Low — lower edge of the Value Area.

    ``bin_price`` is the midpoint of each bin and ``volume`` the total volume
    that traded in that bin (each candle's volume is spread proportionally to
    the overlap of its ``[low, high]`` range with each bin).
    """

    bin_price: np.ndarray  # midpoints of each price bin
    volume: np.ndarray  # volume traded in each bin (same length as bin_price)
    total_volume: float
    poc: float  # Point of Control (midpoint of the max-volume bin)
    poc_volume: float  # volume at the POC
    vah: float  # Value Area High
    val: float  # Value Area Low
    value_volume: float  # total volume inside the value area


@dataclass(frozen=True)
class VolumeProfileSnapshot:
    """One observation's derived levels from an (online) accumulating profile.

    Mirrors the fields of :class:`VolumeProfile` but is produced by the
    incremental ``OnlineVolumeProfile`` accumulator, one candle at a time. The
    scalar levels (``poc`` / ``vah`` / ``val``) are ``None`` until enough
    candles have accumulated to form a profile (``ready=False`` during warmup).
    """

    bin_price: np.ndarray  # midpoints of each price bin
    volume: np.ndarray  # volume traded in each bin (same length as bin_price)
    total_volume: float
    poc: float | None  # Point of Control
    poc_volume: float
    vah: float | None  # Value Area High
    val: float | None  # Value Area Low
    value_volume: float
    n_candles: int
    ready: bool  # False during warmup (insufficient accumulated candles)
