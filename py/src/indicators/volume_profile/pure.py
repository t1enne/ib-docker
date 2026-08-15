"""Pure, stateless computation for the Volume Profile indicator.

``volume_profile`` aggregates the trade volume of a range of candles onto a set
of equal-width price bins, then derives the classic market-structure levels
(Point of Control, Value Area High/Low) from that distribution. It has no side
effects and holds no state — the whole range is profiled in one vectorized call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.volume_profile.types import VolumeProfile


def volume_profile(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Compute a fixed-range Volume Profile over the given candles.

    Prices between the range's ``low`` minimum and ``high`` maximum are split
    into ``num_bins`` equal-width bins. Each candle's volume is distributed
    across the bins its ``[low, high]`` bar overlaps, weighted by overlap
    length, so a wide candle contributes to every level it passed through.

    Args:
        high: High prices for the candles in the range.
        low: Low prices for the candles in the range.
        volume: Trade volume for each candle.
        num_bins: Number of price bins (default 50).
        value_area_pct: Fraction of total volume that defines the Value Area,
            centered on the POC and expanded until this share is covered
            (classic default 0.70).

    Returns:
        A :class:`VolumeProfile` with the per-bin distribution and the POC /
        Value Area levels.

    Raises:
        ValueError: If ``num_bins < 1``, value_area_pct is not in ``(0, 1]``,
            or the candles span a non-positive (empty / NaN) price range.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if not 0.0 < value_area_pct <= 1.0:
        raise ValueError(f"value_area_pct must be in (0, 1], got {value_area_pct}")

    _high = pd.to_numeric(high, errors="coerce")
    _low = pd.to_numeric(low, errors="coerce")
    _vol = pd.to_numeric(volume, errors="coerce")

    # Drop invalid rows (NaN or high < low); keep them out of the distribution.
    valid = _high.notna() & _low.notna() & _vol.notna() & (_high >= _low)
    valid_high = _high[valid]
    valid_low = _low[valid]
    valid_vol = _vol[valid]

    lo = float(np.min(valid_low)) if len(valid_low) else float("nan")
    hi = float(np.max(valid_high)) if len(valid_high) else float("nan")
    if not (hi > lo):
        # Empty range or a single flat price level -> nothing to profile.
        raise ValueError(
            "no valid candles to profile: need at least one non-empty "
            "price range (finite high >= low) and NaN-free volume"
        )

    edges = np.linspace(lo, hi, num_bins + 1)
    bin_lo = edges[:-1]
    bin_hi = edges[1:]
    mids = (bin_lo + bin_hi) / 2.0

    bin_volume = _aggregate_overlap(
        valid_low.values, valid_high.values, valid_vol.values, bin_lo, bin_hi
    )

    return _levels_to_profile(bin_lo, bin_hi, mids, bin_volume, value_area_pct)


def _aggregate_overlap(
    lows: np.ndarray,
    highs: np.ndarray,
    volumes: np.ndarray,
    bin_lo: np.ndarray,
    bin_hi: np.ndarray,
) -> np.ndarray:
    """Distribute each candle's volume across fixed price bins (overlap-weighted).

    For every candle ``[low, high]`` and bin ``[bin_lo, bin_hi]`` the overlap
    length is computed; each candle's volume is then split across its spanned
    bins proportionally to overlap length, so a wide candle contributes to
    every level it passed through. Returns the per-bin volume histogram.
    """
    left = np.maximum(lows[:, None], bin_lo[None, :])
    right = np.minimum(highs[:, None], bin_hi[None, :])
    overlap = np.maximum(0.0, right - left)
    bar_range = (highs - lows).clip(min=1e-12)
    weights = overlap / bar_range[:, None]  # rows sum to 1 for each candle
    # Zero-width (point) candles get zero interval overlap; snap them to their
    # nearest bin so a flat level keeps its volume instead of vanishing.
    point = (highs - lows) <= np.finfo(np.float64).eps
    if np.any(point):
        mids = (bin_lo + bin_hi) / 2.0
        idxs = np.flatnonzero(point)
        for i in idxs:
            center = float(lows[i] + highs[i]) / 2.0
            bin_i = int(np.argmin(np.abs(mids - center)))
            weights[i, :] = 0.0
            weights[i, bin_i] = 1.0
    return (weights * volumes[:, None]).sum(axis=0)


def _candle_overlap(
    low: float, high: float, bin_lo: np.ndarray, bin_hi: np.ndarray
) -> np.ndarray:
    """Overlap weights for a single candle against a fixed bin frame.

    Row sums to 1 when the candle sits fully inside the frame; a candle that
    straddles the frame edge yields a sub-unity (partial-fill) row. Used by the
    online accumulator to add/evict one candle in O(num_bins).
    """
    if high - low <= np.finfo(np.float64).eps:
        # Zero-width (flat) candle: snap its full volume to the nearest bin so
        # it is not silently dropped (interval overlap at a point is ~0).
        mids = (bin_lo + bin_hi) / 2.0
        idx = int(np.argmin(np.abs(mids - (low + high) / 2.0)))
        out = np.zeros(bin_lo.shape[0], dtype=float)
        out[idx] = 1.0
        return out
    left = np.maximum(low, bin_lo)
    right = np.minimum(high, bin_hi)
    overlap = np.maximum(0.0, right - left)
    return overlap / max(high - low, 1e-12)


def _levels_to_profile(
    bin_lo: np.ndarray,
    bin_hi: np.ndarray,
    mids: np.ndarray,
    bin_volume: np.ndarray,
    value_area_pct: float,
) -> VolumeProfile:
    """Derive POC and Value Area levels from a completed bin histogram."""
    total = float(bin_volume.sum())
    poc_idx = int(np.argmax(bin_volume))
    poc = float(mids[poc_idx])

    # Expand the Value Area outward from the POC, one bin at a time, choosing
    # the adjacent bin with the most volume, until we cover value_area_pct.
    included = np.zeros(mids.shape[0], dtype=bool)
    included[poc_idx] = True
    covered = bin_volume[poc_idx]
    left_i = poc_idx
    right_i = poc_idx
    while covered < value_area_pct * total - 1e-9:
        can_left = left_i > 0 and not included[left_i - 1]
        can_right = right_i < mids.shape[0] - 1 and not included[right_i + 1]
        if not can_left and not can_right:
            break  # whole range already included
        if can_left and can_right:
            advance_left = bin_volume[left_i - 1] >= bin_volume[right_i + 1]
        else:
            advance_left = can_left
        if advance_left:
            left_i -= 1
            included[left_i] = True
            covered += bin_volume[left_i]
        else:
            right_i += 1
            included[right_i] = True
            covered += bin_volume[right_i]

    val = float(bin_lo[left_i])
    vah = float(bin_hi[right_i])
    value_volume = float(bin_volume[included].sum())

    return VolumeProfile(
        bin_price=mids,
        volume=bin_volume,
        total_volume=total,
        poc=poc,
        poc_volume=float(bin_volume[poc_idx]),
        vah=vah,
        val=val,
        value_volume=value_volume,
    )
