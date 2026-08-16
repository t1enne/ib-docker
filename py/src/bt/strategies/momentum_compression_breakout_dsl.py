"""Massive-move → pullback → compression → breakout continuation (DSL).

Trades the "flywheel after a big run" pattern on daily bars (resampled from
the native 1h feed):

  1. **Big move (setup)** — the stock gained 30–100% over the trailing 3
     months. Not parabolic: capped at ``max_gain`` so we don't chase a
     blow-off top. Measured against the recent high so a post-run pullback
     doesn't invalidate the setup.
  2. **Pullback + compression** — price pulls back into, and coils between,
     the 10 and 20 SMAs. Compression is a *very small candle body* (a small
     fraction of ATR) on *low volume* while the close hovers at/inside the
     [SMA10, SMA20] band for several consecutive bars. A compression box
     (highest high / lowest low over the window) is retained for the trigger.
  3. **Breakout entry** — long when the close breaks above the box high with
     the 10 SMA above the 20 SMA (uptrend intact).
  4. **Ride the trend** — exit on a manual ATR stop (hard risk cap) or when
     the close drops below a *downward-sloping* 10 SMA (the trend signal
     rolling over).

Cross-candle state (per-symbol compression boxes + cooldowns) lives in
``ctx.shared`` via ``@strategy(stateful=True)`` — the DSL replacement for the
``GLOBAL`` dict, minted fresh per run/window so split/sweep/optimize windows
are isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "momentum_compression_breakout_dsl"

_BOX_KEY = "comp_boxes"  # ctx.shared[sym] -> _Box
_COOLDOWN_KEY = "cooldowns"


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- big-move setup ----------------------------------------------------
    big_lookback: int = 63  # ~3 months of trading days
    min_gain: float = 0.30  # must be up at least 30% over the lookback
    max_gain: float = 1.00  # ...but not parabolic (cap at 100%)
    # -- compression -------------------------------------------------------
    ma_fast: int = 10
    ma_slow: int = 20
    comp_window: int = 15  # look back for the compression box high/low
    body_atr_ratio: float = 0.35  # candle body must be <= this * ATR
    vol_period: int = 20
    vol_mult: float = 0.8  # compression volume <= this * avg volume
    min_hover_bars: int = 3  # consecutive bars close hovering in the MA band
    hover_tol: float = (
        0.002  # close may fall within this fraction of ATR of the band edge
    )
    decay_bars: int = 10  # a detected box stays live for this many bars
    # -- breakout + risk ---------------------------------------------------
    atr_period: int = 14
    atr_mult: float = 2.0  # stop distance = atr_mult * ATR
    risk_pct: float = 0.01  # risk of current cash per trade
    warmup_bars: int = 80
    cooldown_bars: int = 5  # bars to wait after an exit before a re-entry


@dataclass
class _Box:
    """A detected compression coil retained for the breakout trigger.

    ``setup_ok`` latches the big-move precondition the first time it is
    confirmed during the coil, so a later range collapse inside the window
    doesn't invalidate an already-valid setup. ``bar_counts`` is the number of
    consecutive bars since the coil last compressed; it resets each time the
    coil re-validates and is the freshness signal for the breakout trigger.
    """

    high: float
    low: float
    stop_atr: float  # ATR value captured at detection, for the entry stop
    bar_counts: int = 0  # bars elapsed since this coil last compressed
    setup_ok: bool = False  # latched: big-move setup confirmed


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _hovering_between_mas(
    closes: SeriesView,
    sma_fast: float,
    sma_slow: float,
    atr_val: float,
    tol: float,
) -> bool:
    """True when ``close[-1]`` sits at/inside the [SMA_fast, SMA_slow] band.

    The band is the min/max of the two SMAs (a golden-cross means both are
    below price; the point is *price nowhere far outside them*). A small ATR
    tolerance lets the close sit right on an MA edge.
    """
    if sma_fast != sma_fast or sma_slow != sma_slow or not atr_val or np.isnan(atr_val):
        return False
    lo, hi = min(sma_fast, sma_slow), max(sma_fast, sma_slow)
    close = float(closes[-1])
    return (lo - tol * atr_val) <= close <= (hi + tol * atr_val)


def _hovering_streak(
    closes: SeriesView,
    sma_fast: SeriesView,
    sma_slow: SeriesView,
    atr_series: SeriesView,
    params: Params,
) -> int:
    """Number of trailing bars whose close hovers in the fast/slow MA band.

    Counts back from the current bar while the hovering predicate holds.
    """
    n = len(closes)
    if n < params.ma_slow + params.min_hover_bars:
        return 0
    streak = 0
    for i in range(params.min_hover_bars + 8):
        if i >= n:
            break
        atr_back = atr_series[-(i + 1)]
        if not float(atr_back) or np.isnan(float(atr_back)):
            break
        hovering = _hovering_between_mas(
            closes,
            float(sma_fast[-(i + 1)]),
            float(sma_slow[-(i + 1)]),
            float(atr_back),
            params.hover_tol,
        )
        if not hovering:
            break
        streak += 1
    return streak


def _is_compression(o, params: Params, avg_vol: float, atr_val: float) -> bool:
    """Body + volume + MA-hover conditions for a single bar."""
    n = len(o.close)
    if n < params.ma_slow + 2 or np.isnan(atr_val) or atr_val <= 0:
        return False
    body = abs(float(o.close[-1]) - float(o.open[-1]))
    small_body = body <= params.body_atr_ratio * atr_val
    vol_ok = avg_vol == avg_vol and float(o.volume[-1]) <= params.vol_mult * avg_vol
    return bool(small_body and vol_ok)


def _avg_volume(ctx: StrategyContext, sym: str, params: Params) -> float:
    """Mean volume over the prior ``vol_period`` bars (excl. current bar)."""
    vol = ctx.ohlcv(sym).volume.to_array()
    n = len(vol)
    if n < params.vol_period + 1:
        return float("nan")
    return float(np.mean(vol[-params.vol_period : -1]))


def _big_move_ok(closes: SeriesView, params: Params) -> bool:
    """Range of the trailing ``big_lookback`` bars is within [min_gain, max_gain].

    The setup is a *completed* run: over the window the price advanced 30–100%
    end to end (peak-over-trough). A tight real-world coil therefore keeps the
    measured range intact (the trough is the pre-run low, the peak the run top)
    regardless of exactly where in the window the pullback lands. The >100%
    parabolic top is excluded. Price currently pulled back into the MA band is
    enforced separately by the hovering + compression conditions (and, at
    entry, the ``fast-SMA > slow-SMA`` uptrend gate).
    """
    n = len(closes)
    if n < params.big_lookback:
        return False
    arr = closes.to_array()[-params.big_lookback :]
    low = float(np.min(arr))
    if low <= 0:
        return False
    high = float(np.max(arr))
    gain = high / low - 1.0
    return params.min_gain <= gain <= params.max_gain


def _coil_conditions(ctx: StrategyContext, sym: str, params: Params) -> bool:
    """Compression (tiny body + low volume) plus the MA-hovering streak."""
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < params.warmup_bars:
        return False
    atr_val = float(ctx.ta.atr(sym, params.atr_period)[-1])
    avg_vol = _avg_volume(ctx, sym, params)
    if not _is_compression(o, params, avg_vol, atr_val):
        return False
    streak = _hovering_streak(
        o.close,
        ctx.ta.sma(sym, params.ma_fast),
        ctx.ta.sma(sym, params.ma_slow),
        ctx.ta.atr(sym, params.atr_period),
        params,
    )
    return streak >= params.min_hover_bars


def _update_box(
    ctx: StrategyContext, sym: str, params: Params, box: _Box | None
) -> _Box | None:
    """Return the armed coil box for the current bar, or ``None`` to disarm.

    The coil stays armed while price remains *coiling*:

      * a compressive (tiny-body, low-volume, MA-hover) bar recomputes the
        box coordinates, resets the freshness counter and latches the big-move
        precondition the first time it is confirmed;
      * a non-compressive bar still inside the existing ``[low, high]`` band
        keeps the box armed (freshness reset) — the coil is intact even if the
        bar isn't itself the tightest;
      * price leaving the band returns ``None`` so the caller disarms it
        (bar_counts ages); the caller should also age a box that sits idle.

    Returns the freshly-computed box (compressive bar) or the existing box
    zeroed back to fresh (in-band bar). ``None`` when price broke the band or
    the coil never formed.
    """
    o = ctx.ohlcv(sym)
    if _coil_conditions(ctx, sym, params):
        # A true compression bar: recompute the box over the window.
        low = float(np.min(o.low.to_array()[-params.comp_window :]))
        high = float(np.max(o.high.to_array()[-params.comp_window :]))
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            return None
        atr_val = float(ctx.ta.atr(sym, params.atr_period)[-1])
        setup_ok = bool(_big_move_ok(o.close, params)) or (
            box is not None and box.setup_ok
        )
        return _Box(
            high=high, low=low, stop_atr=atr_val, bar_counts=0, setup_ok=setup_ok
        )
    if isinstance(box, _Box):
        # Not a compressive bar, but is price still coiling inside the band?
        close = float(o.close[-1])
        if box.low <= close <= box.high:
            return box  # keep armed: the coil is intact
    return None


def _exit_reason(
    ctx: StrategyContext, sym: str, params: Params, entry_price: float
) -> tuple[str, float | None] | None:
    """Manual exits: ATR stop (guard) or close < slowing-down 10 SMA.

    Returns ``(reason, stop_guard)``; ``stop_guard`` is None for the trend
    exit (fill at next open) and the stop level for the ATR exit.
    """
    o = ctx.ohlcv(sym)
    close = float(o.close[-1])
    atr_val = float(ctx.ta.atr(sym, params.atr_period)[-1])
    stop_dist = (
        params.atr_mult * atr_val
        if atr_val > 0 and not np.isnan(atr_val)
        else float("inf")
    )
    sma_f = float(ctx.ta.sma(sym, params.ma_fast)[-1])
    sma_f_prev = float(ctx.ta.sma(sym, params.ma_fast)[-2])
    if float(o.low[-1]) <= entry_price - stop_dist:
        return ("[stop] ATR stop hit", entry_price - stop_dist)
    if sma_f != sma_f or sma_f_prev != sma_f_prev:
        return None
    if close < sma_f and sma_f < sma_f_prev:
        return ("[trend] close below sloping-down 10 SMA", None)
    return None


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_COOLDOWN_KEY, {})

    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        lots = ctx.position_ids(sym)

        # ---- exits --------------------------------------------------------
        if lots:
            pos = ctx.position(sym)
            assert pos is not None
            outcome = _exit_reason(ctx, sym, params, float(pos.entry_price))
            if outcome:
                reason, guard = outcome
                if guard is None:
                    ctx.close(sym, reason=reason)
                else:
                    ctx.close(sym, reason=reason, guard_price=guard)
                shared[_COOLDOWN_KEY][sym] = params.cooldown_bars
            continue

        # ---- cooldown after an exit ---------------------------------------
        cd = shared[_COOLDOWN_KEY].get(sym, 0)
        if cd > 0:
            shared[_COOLDOWN_KEY][sym] = cd - 1
            continue

        # ---- detect / refresh the compressive coil + breakout entry ---------
        boxes = shared.setdefault(_BOX_KEY, {})
        box = boxes.get(sym)

        fresh = _update_box(ctx, sym, params, box)
        if fresh is not None:
            boxes[sym] = fresh
            box = fresh
        elif isinstance(box, _Box):
            # Price left the band (breakout or breakdown): age the coil so an
            # idle setup eventually disarms; the breakout check below still
            # runs this bar against the established box high.
            box.bar_counts += 1

        live = (
            box
            if (
                isinstance(box, _Box)
                and box.setup_ok
                and box.bar_counts <= params.decay_bars
            )
            else None
        )
        if live is None:
            continue

        close = float(o.close[-1])
        sma_f = float(ctx.ta.sma(sym, params.ma_fast)[-1])
        sma_s = float(ctx.ta.sma(sym, params.ma_slow)[-1])
        if sma_f != sma_f or sma_s != sma_s:
            continue
        if not (close > live.high and sma_f > sma_s):
            continue  # breakout must be real and trend must be intact

        # ATR-risk sizing (strategy-owned): back-solve ctx.long's 0..1 size
        # from the absolute share count that risks ``risk_pct`` of cash.
        price = close
        cash = ctx.state.portfolio.cash
        if price <= 0 or cash <= 0 or live.stop_atr <= 0 or np.isnan(live.stop_atr):
            continue
        stop_dist = params.atr_mult * live.stop_atr
        qty = risk_sized_qty(
            equity=cash, price=price, stop_dist=stop_dist, risk_pct=params.risk_pct
        )
        if qty <= 0:
            continue
        size = qty * price / ctx.state.portfolio.initial_capital
        size = min(max(size, 0.0), 1.0)
        if size <= 0:
            continue

        ctx.long(
            sym,
            size=size,
            reason=f"[breakout] close {close:.2f} > box high {live.high:.2f}",
        )
        # Consume the box so an immediate re-breakout can't re-enter flat.
        boxes.pop(sym, None)
