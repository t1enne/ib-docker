"""Cup and Handle breakout — DSL-native, self-contained port.

Same economic logic as the removed ``cup_handle`` strategy (pure geometric
detector + breakout entry + ratcheting ATR trail), expressed on the
declarative :class:`StrategyContext` DSL.

* The pure geometric detector + trend helpers (``detect_cup_and_handle``,
  ``cap_stop_dist``, ``per_symbol_qty``, ``is_uptrend``, ``uptrend_aligned``,
  ``is_bull_market``) are inlined here so the module is self-contained.
* Data is read cursor-safe through ``ctx.ohlcv`` (cursor-truncated numpy
  arrays -> ``pd.Series`` for the pure detector) and ``ctx.ta.atr`` (one
  full-series ATR compute, O(1) per-candle reads).
* Cross-candle state (cooldowns, handle_lows, trail_stops) lives in
  ``ctx.shared`` via ``@strategy(stateful=True)`` — the DSL replacement for
  the ``GLOBAL`` dict.

Sizing / SL-TP semantics (DSL convention):

* ``ctx.long(size=...)`` sizes a fraction of *initial* capital, i.e.
  ``qty = size * initial_capital / price``. Passing ``size=per_symbol_size``
  reproduces the original's ``per_symbol_qty(initial_capital, per_symbol_size,
  price)`` exactly.
* ``ctx.long(sl=, tp=)`` interprets its args as *fractional percentages* of
  entry converted via ``sl_tp_from_pct`` (absolute = entry*(1±pct)), NOT as
  absolute prices. The original computes absolute ``stop_loss``/``target``
  levels, so those are converted back to fractions before emission:
  ``sl_pct = stop_dist / entry`` and ``tp_pct = (target - entry) / entry``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams


STRATEGY_TYPE = "cup_handle_dsl"


# ---------------------------------------------------------------------------
# pure geometric detector + trend helpers (cursor-safe inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Swing:
    """A single turning point on the price series."""

    idx: int
    high: bool  # True = pivot high, False = pivot low
    level: float


@dataclass(frozen=True)
class Handle:
    """A recognized handle off a cup's right rim."""

    start_idx: int  # pivot-high index = the resistance/breakout line
    low_idx: int  # index of the handle's lowest bar
    low: float
    breakout_level: float  # sell above this
    cup_depth: float  # absolute price depth of the parent cup
    rim_level: float  # right-rim price (for reference / target calc)


@dataclass(frozen=True)
class CupHandleResult:
    cup_ok: bool
    handle_ok: bool
    entry_ok: bool
    handle: Optional[Handle]
    reason: str


def per_symbol_qty(
    initial_capital: float, per_symbol_size: float, price: float
) -> float:
    """Shares for a fixed ``per_symbol_size`` fraction of total capital."""
    if initial_capital <= 0 or price <= 0 or per_symbol_size <= 0:
        return 0.0
    return round(initial_capital * per_symbol_size / price, 4)


def cap_stop_dist(
    natural_dist: float,
    risk_cap_dollars: float,
    qty: float,
    *,
    atr_val: float,
    min_stop_atr: float = 0.5,
) -> float:
    """Bounded stop distance so a long's stop-out can never over-lose."""
    cap_dist = risk_cap_dollars / qty if qty > 0 else float("inf")
    dist = min(natural_dist, cap_dist)
    return max(dist, min_stop_atr * atr_val)


def trend_strength(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    *,
    adx_window: int = 14,
) -> float:
    """Latest ADX reading (0..100); >= ``adx_trend_threshold`` = trending."""
    adx_series = ta.adx(highs, lows, closes, adx_window)
    val = float(adx_series.iloc[-1])
    return val if val == val else 0.0  # NaN guard


def is_uptrend(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    *,
    adx_window: int = 14,
    threshold: float = 25.0,
    ma_span: int = 50,
) -> bool:
    """Trending up? ADX >= threshold AND close above a slow moving average."""
    if len(closes) < ma_span + 1:
        return False
    adx_val = trend_strength(highs, lows, closes, adx_window=adx_window)
    if adx_val < threshold:
        return False
    ma = float(ta.sma(closes, ma_span).iloc[-1])
    return float(closes.iloc[-1]) > ma


def is_bull_market(
    closes: pd.Series,
    *,
    ma_span: int = 200,
    slope_span: int = 20,
    min_slope_frac: float = 0.0,
) -> bool:
    """Is the broad market in a bullish regime (price above a long MA)?"""
    if len(closes) < ma_span + slope_span:
        return False
    ma = ta.sma(closes, ma_span)
    last_ma = float(ma.iloc[-1])
    if last_ma != last_ma or last_ma <= 0:  # NaN guard
        return False
    if min_slope_frac > 0:
        past_ma = float(ma.iloc[-slope_span - 1])
        if past_ma != past_ma:  # NaN guard
            return False
        slope = (last_ma - past_ma) / past_ma
        if slope < min_slope_frac:
            return False
    return float(closes.iloc[-1]) > last_ma


def uptrend_aligned(
    sym_close: pd.Series,
    mkt_close: pd.Series,
    *,
    sym_span: int = 100,
    mkt_span: int = 200,
) -> bool:
    """Both symbol and market proxy close above their own long MA."""
    return is_bull_market(
        sym_close, ma_span=sym_span, slope_span=20, min_slope_frac=0.0
    ) and is_bull_market(mkt_close, ma_span=mkt_span, slope_span=20, min_slope_frac=0.0)


def _find_swings(highs: pd.Series, lows: pd.Series, lookback: int) -> list[Swing]:
    """Return pivot highs/lows using a symmetric fractal window."""
    n = len(highs)
    if n < 2 * lookback + 1:
        return []

    swings: list[Swing] = []
    highs_np = highs.to_numpy()
    lows_np = lows.to_numpy()
    for i in range(lookback, n - lookback):
        lo, hi = i - lookback, i + lookback
        hi_max = float(highs_np[lo:i].max()) if lo < i else float(highs_np[i])
        hi_max = max(
            hi_max,
            float(highs_np[i + 1 : hi + 1].max()) if i + 1 <= hi else -float("inf"),
        )
        lo_min = float(lows_np[lo:i].min()) if lo < i else float(lows_np[i])
        lo_min = min(
            lo_min,
            float(lows_np[i + 1 : hi + 1].min()) if i + 1 <= hi else float("inf"),
        )
        if highs_np[i] > hi_max:
            swings.append(Swing(i, True, float(highs_np[i])))
        elif lows_np[i] < lo_min:
            swings.append(Swing(i, False, float(lows_np[i])))
    return swings


def _check_cup(
    swings: Sequence[Swing],
    left: Swing,
    bottom: Swing,
    right: Swing,
    *,
    max_cup_depth_pct: float,
    min_cup_depth_pct: float,
    rim_tolerance_pct: float,
    min_mid_pivots: int,
) -> bool:
    """Validate a cup's geometry between a left rim, a bottom low and a right rim."""
    depth = (left.level - bottom.level) / left.level
    if not (min_cup_depth_pct <= depth <= max_cup_depth_pct):
        return False
    if not (left.idx < bottom.idx < right.idx):
        return False
    if abs(right.level - left.level) / left.level > rim_tolerance_pct:
        return False

    mid = [s for s in swings if bottom.idx < s.idx < right.idx]
    if len(mid) < min_mid_pivots:
        return False
    after_bottom_lows = [s for s in mid if not s.high]
    if not any(s.level > bottom.level for s in after_bottom_lows):
        return False  # no upturn on the right wall -> V bottom
    return True


def _find_handle(
    swings: Sequence[Swing],
    right: Swing,
    n: int,
    *,
    max_handle_bars: int,
    max_handle_drop_pct: float,
) -> Optional[Handle]:
    """Locate the small pullback (handle) immediately following the right rim."""
    end = min(right.idx + max_handle_bars, n)
    seg = [s for s in swings if right.idx < s.idx < end]

    seg_lows = [s for s in seg if not s.high]
    if not seg_lows:
        return None
    handle_low = min(seg_lows, key=lambda s: s.level)

    drop = (right.level - handle_low.level) / right.level
    if drop > max_handle_drop_pct:
        return None

    return Handle(
        start_idx=right.idx,
        low_idx=handle_low.idx,
        low=handle_low.level,
        breakout_level=right.level,
        cup_depth=0.0,  # filled by the caller
        rim_level=right.level,
    )


def _volume_picks_up(volumes: pd.Series, rim_idx: int, n: int) -> bool:
    """Breakout bar volume should exceed the cup-body average."""
    vol = volumes.to_numpy()
    if n < 2 or vol[-1] != vol[-1]:  # NaN guard
        return False
    base = vol[rim_idx : n - 1]
    if len(base) == 0:
        return False
    return float(vol[-1]) > float(np.mean(base))


def detect_cup_and_handle(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: Optional[pd.Series],
    *,
    swing_lookback: int = 3,
    max_cup_depth_pct: float = 0.40,
    min_cup_depth_pct: float = 0.10,
    rim_tolerance_pct: float = 0.05,
    min_mid_pivots: int = 1,
    max_cup_bars: int = 260,
    max_handle_bars: int = 40,
    max_handle_drop_pct: float = 0.12,
    volume_confirm_breakout: bool = True,
) -> CupHandleResult:
    """Detect a cup-and-handle formation ending at (breaking on) the last bar."""
    n = len(highs)
    if not (len(lows) == n and len(closes) == n):
        return CupHandleResult(False, False, False, None, "length mismatch")
    if volumes is not None and len(volumes) != n:
        return CupHandleResult(False, False, False, None, "volume length mismatch")
    if n < 2 * swing_lookback + 1:
        return CupHandleResult(False, False, False, None, "insufficient data")

    swings = _find_swings(highs, lows, swing_lookback)
    closes_np = closes.to_numpy()
    last_close = float(closes_np[-1])

    best: Optional[Handle] = None
    cup_found = False

    # Iterate candidate right rims from most recent backwards.
    for j in range(len(swings) - 1, -1, -1):
        right = swings[j]
        if not right.high:
            continue
        if right.idx > n - 1 - swing_lookback - 1:
            continue
        for i in range(j - 1, -1, -1):
            left = swings[i]
            if not left.high:
                continue
            if right.idx - left.idx > max_cup_bars:
                break
            for b in range(i + 1, j):
                bottom = swings[b]
                if bottom.high:
                    continue
                if not _check_cup(
                    swings,
                    left,
                    bottom,
                    right,
                    max_cup_depth_pct=max_cup_depth_pct,
                    min_cup_depth_pct=min_cup_depth_pct,
                    rim_tolerance_pct=rim_tolerance_pct,
                    min_mid_pivots=min_mid_pivots,
                ):
                    continue

                cup_found = True
                depth = left.level - bottom.level
                handle = _find_handle(
                    swings,
                    right,
                    n,
                    max_handle_bars=max_handle_bars,
                    max_handle_drop_pct=max_handle_drop_pct,
                )
                if handle is None:
                    continue
                handle = Handle(
                    start_idx=handle.start_idx,
                    low_idx=handle.low_idx,
                    low=handle.low,
                    breakout_level=handle.breakout_level,
                    cup_depth=depth,
                    rim_level=right.level,
                )

                if best is None or right.idx > best.start_idx:
                    best = handle

    if best is None:
        reason = "handle" if cup_found else "no valid cup"
        return CupHandleResult(cup_found, False, False, None, f"no {reason}")

    if last_close <= best.breakout_level:
        return CupHandleResult(True, True, False, best, "no breakout yet")

    if volume_confirm_breakout and volumes is not None:
        if not _volume_picks_up(volumes, best.start_idx, n):
            return CupHandleResult(True, True, False, best, "no volume confirmation")

    return CupHandleResult(True, True, True, best, "breakout")


@dataclass(frozen=True)
class Params(StrategyParams):
    # Swing detection
    swing_lookback: int = 3
    # Cup geometry
    max_cup_depth_pct: float = 0.40
    min_cup_depth_pct: float = 0.10
    rim_tolerance_pct: float = 0.05
    min_mid_pivots: int = 1
    max_cup_bars: int = 260
    # Handle geometry
    max_handle_bars: int = 40
    max_handle_drop_pct: float = 0.12
    # Volume confirmation
    volume_confirm_breakout: bool = True
    # Sizing / risk
    per_symbol_size: float = 0.30
    max_risk_pct: float = 0.02
    min_stop_atr: float = 0.5
    # Exit / trail
    trail_atr_period: int = 14
    trail_atr_mult: float = 2.5
    adx_window: int = 14
    adx_trend_threshold: float = 25.0
    trend_trail_atr_mult: float = 3.0
    chop_trail_atr_mult: float = 2.0
    trend_tp_mult: float = 1.2
    # Warmup & cooldown
    warmup_bars: int = 60
    cooldown_bars: int = 5
    profit_target_mult: float = 1.0


# ---------------------------------------------------------------------------
# data access helpers (cursor-safe; build pd.Series for the pure detector)
# ---------------------------------------------------------------------------


def _arrays(ctx: StrategyContext, sym: str) -> dict[str, pd.Series]:
    """Cursor-truncated OHLCV as pandas Series (matches pure fn inputs)."""
    o = ctx.ohlcv(sym)
    return {
        "close": pd.Series(o.close.to_array()),
        "high": pd.Series(o.high.to_array()),
        "low": pd.Series(o.low.to_array()),
        "volume": pd.Series(o.volume.to_array()),
    }


def _spy_closes(ctx: StrategyContext) -> pd.Series | None:
    """SPY daily closes (broad-market proxy), or None when SPY isn't traded."""
    if "SPY" not in ctx.symbols:
        return None
    arr = ctx.ohlcv("SPY").close.to_array()
    return pd.Series(arr) if len(arr) > 0 else None


def _market_regime_ok(ctx: StrategyContext) -> bool:
    """New entries only when a broad-market bull gate is open (SPY > 200MA)."""
    spy = _spy_closes(ctx)
    return True if spy is None else bool(is_bull_market(spy))


def _symbol_market_aligned(ctx: StrategyContext, closes: pd.Series) -> bool:
    """Symbol + SPY long-MA uptrend alignment (level-based)."""
    spy = _spy_closes(ctx)
    return True if spy is None else bool(uptrend_aligned(closes, spy))


def _atr_val(ctx: StrategyContext, sym: str, period: int) -> float:
    v = float(ctx.ta.atr(sym, period)[-1])
    return v if not np.isnan(v) and v > 0 else 0.0


# ---------------------------------------------------------------------------
# entry / position-management
# ---------------------------------------------------------------------------


def _maybe_enter(ctx: StrategyContext, sym: str, arr: dict[str, pd.Series]) -> None:
    params = ctx.params
    closes = arr["close"]
    if len(closes) < params.warmup_bars:
        return

    result = detect_cup_and_handle(
        arr["high"],
        arr["low"],
        closes,
        arr["volume"],
        swing_lookback=params.swing_lookback,
        max_cup_depth_pct=params.max_cup_depth_pct,
        min_cup_depth_pct=params.min_cup_depth_pct,
        rim_tolerance_pct=params.rim_tolerance_pct,
        min_mid_pivots=params.min_mid_pivots,
        max_cup_bars=params.max_cup_bars,
        max_handle_bars=params.max_handle_bars,
        max_handle_drop_pct=params.max_handle_drop_pct,
        volume_confirm_breakout=params.volume_confirm_breakout,
    )
    if not result.entry_ok or result.handle is None:
        return
    handle = result.handle

    if not _symbol_market_aligned(ctx, closes):
        return

    entry_price = float(closes.iloc[-1])

    atr_val = _atr_val(ctx, sym, params.trail_atr_period)
    if atr_val <= 0:
        atr_val = max(handle.cup_depth * 0.1, 1e-9)

    initial = ctx.state.portfolio.initial_capital
    qty = per_symbol_qty(initial, params.per_symbol_size, entry_price)
    if qty <= 0 or ctx.state.portfolio.cash <= 0:
        return

    natural_dist = max(
        params.trail_atr_mult * atr_val, (entry_price - handle.low) * 1.05
    )
    risk_cap_dollars = initial * params.max_risk_pct
    stop_dist = cap_stop_dist(
        natural_dist,
        risk_cap_dollars,
        qty,
        atr_val=atr_val,
        min_stop_atr=params.min_stop_atr,
    )

    trending = is_uptrend(
        arr["high"],
        arr["low"],
        closes,
        adx_window=params.adx_window,
        threshold=params.adx_trend_threshold,
    )
    target = 0.0
    if not trending and params.trend_tp_mult > 0 and handle.cup_depth > 0:
        target = entry_price + params.trend_tp_mult * handle.cup_depth

    # Convert absolute SL/TP levels back to fractional pct for ctx.long.
    sl_pct = stop_dist / entry_price if stop_dist > 0 else 0.0
    tp_pct = (target - entry_price) / entry_price if target > 0 else 0.0

    ctx.shared["handle_lows"][sym] = handle.low
    ctx.shared["cooldowns"][sym] = params.cooldown_bars
    ctx.long(
        sym,
        size=params.per_symbol_size,
        sl=sl_pct,
        tp=tp_pct,
        reason=(
            f"[cup&handle] breakout above {handle.breakout_level:.2f} "
            f"(depth={handle.cup_depth:.2f} · {result.reason})"
        ),
    )


def _manage_position(ctx: StrategyContext, sym: str, arr: dict[str, pd.Series]) -> None:
    pos = ctx.position(sym)
    if pos is None:
        return
    # Chop entries carry a fixed TP; the engine's TP/SL handles those — the
    # ratchet is reserved for trend (no-TP) entries.
    if pos.take_profit is not None:
        return

    params = ctx.params
    closes = arr["close"]
    close_price = float(closes.iloc[-1])

    handle_low = ctx.shared["handle_lows"].get(sym)
    atr_val = _atr_val(ctx, sym, params.trail_atr_period)
    if atr_val <= 0:
        atr_val = max(float(closes.iloc[-1]) * 0.01, 1e-9)

    trending = is_uptrend(
        arr["high"],
        arr["low"],
        closes,
        adx_window=params.adx_window,
        threshold=params.adx_trend_threshold,
    )
    trail_mult = params.trend_trail_atr_mult if trending else params.chop_trail_atr_mult

    pid = pos.position_id
    candidate = close_price - trail_mult * atr_val
    floor = handle_low * 0.99 if handle_low else -float("inf")
    seed = float(pos.stop_loss) if pos.stop_loss else -float("inf")
    base = max(candidate, floor, seed)
    saved = ctx.shared["trail_stops"].get(pid)
    trail_stop = base if saved is None else max(base, saved)
    ctx.shared["trail_stops"][pid] = trail_stop

    if close_price > trail_stop:
        return

    ctx.shared["trail_stops"].pop(pid, None)
    ctx.shared["cooldowns"][sym] = params.cooldown_bars
    ctx.close(
        sym,
        reason=(
            f"[cup&handle] ratchet trail hit "
            f"(close {close_price:.2f} <= {trail_stop:.2f})"
        ),
    )


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    shared = ctx.shared
    shared.setdefault("cooldowns", {})
    shared.setdefault("handle_lows", {})
    shared.setdefault("trail_stops", {})

    symbols = sorted(ctx.symbols)

    # Bull-market gate, evaluated lazily: skip the O(n) SMA work entirely when
    # every symbol already holds a position (no possible new entry this bar).
    can_enter = any(ctx.position(sym) is None for sym in symbols)
    bull_ok = True if not can_enter else _market_regime_ok(ctx)

    for sym in symbols:
        if ctx.position(sym) is not None:
            _manage_position(ctx, sym, _arrays(ctx, sym))
            continue

        cd = shared["cooldowns"].get(sym, 0)
        if cd > 0:
            shared["cooldowns"][sym] = cd - 1
            continue

        if not bull_ok:
            continue
        _maybe_enter(ctx, sym, _arrays(ctx, sym))
