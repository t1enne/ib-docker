"""Cup and Handle breakout — pure geometric detector + on_candle strategy.

Cup and Handle is a price-structure pattern, so this module derives it
directly from OHLCV bars (no images / no fixed oscillators):

  SWINGS  ->  CUP (left rim -> rounded bottom -> right rim)  ->  HANDLE
  (small pullback off the right rim)  ->  BREAKOUT (close above the
  handle's pivot high), with optional volume confirmation.

All pattern logic lives in the PURE functions (_find_swings,
detect_cup_and_handle, ...) so the geometry is unit-testable in isolation.
on_candle() is a thin engine adapter over them.

Convention: cup is measured in bars. Use daily bars for the classic
1-6 month cup; the geometry params are all in fractions of price.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "cup_handle"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    # Swing detection
    swing_lookback: int = 3  # bars each side to confirm a pivot

    # Cup geometry (fractions of price; all 0..1)
    max_cup_depth_pct: float = 0.40  # (rim-bottom)/rim upper bound
    min_cup_depth_pct: float = 0.10  # ignore saucers shallower than this
    rim_tolerance_pct: float = 0.05  # right rim within this % of left rim
    min_mid_pivots: int = 1  # min right-wall pullbacks in the base ("U" not "V")
    max_cup_bars: int = 260  # ~1 year of daily bars; hard width cap

    # Handle geometry
    max_handle_bars: int = 40  # handle must be much shorter than the cup
    max_handle_drop_pct: float = 0.12  # handle pullback off the right rim

    # Volume confirmation (optional)
    volume_confirm_breakout: bool = True  # require vol pick-up on breakout

    # Sizing / risk: fixed fraction of TOTAL capital (initial_capital) per
    # symbol, so each symbol always deploys the same allocation regardless of
    # how many symbols trade simultaneously or current cash level.
    per_symbol_size: float = 0.30

    # Hard cap on loss per trade, as a fraction of the account. The stop-loss
    # is set so a stop-out loses at most ``max_risk_pct`` of capital, flipping
    # the per-trade risk/reward from >1 (risk larger than reward) to <1.
    max_risk_pct: float = 0.02

    # Stop never tighter than this many ATRs below entry (anti-whipsaw floor).
    min_stop_atr: float = 0.5

    # Exit / trail
    trail_atr_period: int = 14
    trail_atr_mult: float = 2.5

    # Trend-aware exits: a fixed price target caps winners inside a strong
    # trend; the ATR trail is what should collect a runner. ADX measures
    # trend strength at entry and during management so we can (a) skip the
    # hard TP in trending regimes and (b) widen/ tighten the trail with the
    # trend.
    adx_window: int = 14
    adx_trend_threshold: float = 25.0  # ADX >= this => trending regime
    # Trail multiple applied when trending (wider => let the move breathe) vs
    # when choppy (tighter => protect gains).
    trend_trail_atr_mult: float = 3.0
    chop_trail_atr_mult: float = 2.0
    # Target multiplier used ONLY in non-trending regimes (<=0 = never)
    trend_tp_mult: float = 1.2

    # Warmup & cooldown
    warmup_bars: int = 60
    cooldown_bars: int = 5

    # Target: multiple of cup depth above the breakout (<=0 = no hard TP).
    # Superseded by the adaptive rule above; kept for parity/back-compat.
    profit_target_mult: float = 1.0


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

GLOBAL: dict = {"cooldowns": {}, "handle_lows": {}, "trail_stops": {}}


def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldowns": {}, "handle_lows": {}, "trail_stops": {}}


# ---------------------------------------------------------------------------
# pure trend helpers
# ---------------------------------------------------------------------------


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
    """Trending up? ADX >= threshold AND close above a slow moving average.

    Combines directional strength (ADX) with bullish bias (price above the
    slow MA) so a high-ADX *down* trend is never treated as a trend we want
    to hold a long through.
    """
    if len(closes) < ma_span + 1:
        return False
    adx_val = trend_strength(highs, lows, closes, adx_window=adx_window)
    if adx_val < threshold:
        return False
    ma = float(ta.sma(closes, ma_span).iloc[-1])
    return float(closes.iloc[-1]) > ma


# ---------------------------------------------------------------------------
# pure pattern detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Swing:
    """A single turning point on the price series."""

    idx: int
    high: bool  # True = pivot high, False = pivot low
    level: float


def per_symbol_qty(
    initial_capital: float, per_symbol_size: float, price: float
) -> float:
    """Shares for a fixed ``per_symbol_size`` fraction of total capital.

    Sizes against the account's starting capital (not current cash) so every
    symbol deploys the same intended allocation regardless of how many
    symbols trade concurrently or how much cash is free at entry time.
    """
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
    """Bounded stop distance for a long so a stop-out never loses too much.

    Takes the tighter of the structure-based ``natural_dist`` and the
    risk-budget distance (``risk_cap_dollars / qty``), floored at
    ``min_stop_atr`` ATRs so the stop never trips on plain noise.
    """
    cap_dist = risk_cap_dollars / qty if qty > 0 else float("inf")
    dist = min(natural_dist, cap_dist)
    return max(dist, min_stop_atr * atr_val)


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


def _find_swings(highs: pd.Series, lows: pd.Series, lookback: int) -> list[Swing]:
    """Return pivot highs/lows using a symmetric fractal window.

    A bar i is a pivot high iff highs[i] is >= every other high in
    [i-lookback, i+lookback]; a pivot low analogously on lows[i].
    Only fully-windowed bars are classified (i in [lookback, n-lookback))
    so live trading has no lookahead: the trailing `lookback` bars are
    left unclassified, as intended.
    """
    n = len(highs)
    if n < 2 * lookback + 1:
        return []

    swings: list[Swing] = []
    highs_np = highs.to_numpy()
    lows_np = lows.to_numpy()
    for i in range(lookback, n - lookback):
        lo, hi = i - lookback, i + lookback
        # Strict turning point: i must beat BOTH sides of its window
        # (excluding itself), so flat plateaus never produce pivots.
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
    """Validate cup geometry between a left rim, a bottom low and a right rim.

    Conditions:
      1. Depth (rim-bottom)/rim within [min_depth, max_depth].
      2. Bottom low sits strictly between the two rims.
      3. Right rim is roughly level with the left rim (tolerance).
      4. Rounded base: at least `min_mid_pivots` pivots sit between the
         bottom and the right rim, and at least one higher-low pivot
         appears after the bottom (a "U", not a right-angle "V").
    """
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
    """Locate the small pullback (handle) immediately following the right rim.

    A handle is the set of bars after `right` whose index is bounded by the
    next pivot high (the handle's own resistance origin). The handle's low
    must stay within max_handle_drop_pct of the right rim and must not exceed
    max_handle_bars in width. Returns None if no handle forms yet.
    """
    end = min(right.idx + max_handle_bars, n)
    seg = [s for s in swings if right.idx < s.idx < end]

    # The handle's low: the lowest swing-low within the segment, but never
    # deeper than the handle drop limit below the rim.
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
    """Detect a cup-and-handle formation ending at (and breaking on) the last bar.

    Returns handle_ok=True when a cup+handle formation exists (geometry
    only). Returns entry_ok=True only when the formation is actionable:
      - a valid cup exists (left rim -> bottom -> right rim),
      - a valid handle has formed off the right rim,
      - the latest close trades above the handle's breakout level,
      - (optionally) breakout volume picks up (volume_confirm_breakout).

    Pure: no engine state, no lookahead beyond naturally-confirmed pivots.
    """
    n = len(highs)
    if not (len(lows) == n and len(closes) == n):
        return CupHandleResult(False, False, False, None, "length mismatch")
    if volumes is not None and len(volumes) != n:
        return CupHandleResult(False, False, False, None, "volume length mismatch")
    if n < 2 * swing_lookback + 1:
        return CupHandleResult(False, False, False, None, "insufficient data")

    swings = _find_swings(highs, lows, swing_lookback)
    closes_np = closes.to_numpy()

    # Latest bar's close (the live breakout probe).
    last_close = float(closes_np[-1])

    best: Optional[Handle] = None
    cup_found = False

    # Iterate candidate right rims from most recent backwards.
    for j in range(len(swings) - 1, -1, -1):
        right = swings[j]
        if not right.high:
            continue
        # Must leave room to confirm the right rim AND a handle beyond it.
        if right.idx > n - 1 - swing_lookback - 1:
            continue
        # Find the left rim and bottom for this right rim.
        for i in range(j - 1, -1, -1):
            left = swings[i]
            if not left.high:
                continue
            if right.idx - left.idx > max_cup_bars:
                break  # left is too far away; stop scanning earlier
            # Candidate bottom: a swing low between left and right.
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

                # Prefer the most recent valid formation.
                if best is None or right.idx > best.start_idx:
                    best = handle

    if best is None:
        reason = "handle" if cup_found else "no valid cup"
        return CupHandleResult(cup_found, False, False, None, f"no {reason}")

    # A valid cup+handle formation exists regardless of breakout status;
    # this keeps ``handle_ok`` a pure "pattern formed" signal. "Ready to
    # enter" additionally requires the close to have broken above the
    # breakout line (and, when enabled, volume confirmation).
    if last_close <= best.breakout_level:
        return CupHandleResult(True, True, False, best, "no breakout yet")

    if volume_confirm_breakout and volumes is not None:
        if not _volume_picks_up(volumes, best.start_idx, n):
            return CupHandleResult(True, True, False, best, "no volume confirmation")

    return CupHandleResult(True, True, True, best, "breakout")


def _volume_picks_up(volumes: pd.Series, rim_idx: int, n: int) -> bool:
    """Breakout bar volume should exceed the cup-body average.

    Compares the latest-bar volume against the mean volume between the
    left rim and the handle, i.e. the quiet base. A clean breakout tends
    to come on 1.5x+ base volume; we use a modest >1.0x filter.
    """
    vol = volumes.to_numpy()
    if n < 2 or vol[-1] != vol[-1]:  # NaN guard
        return False
    base = vol[rim_idx : n - 1]
    if len(base) == 0:
        return False
    return float(vol[-1]) > float(np.mean(base))


# ---------------------------------------------------------------------------
# on_candle — thin engine adapter over the pure detector
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []

    # Only trade on the daily resolution. HTF/base candles of other
    # intervals never drive decisions for this strategy.
    interval = candle.interval or "1d"
    if interval != "1d":
        return signals

    symbols: list[str] = sorted({k[0] for k in state.candles.keys()})
    for symbol in symbols:
        pos_tup = state.portfolio.positions.get(symbol, ())
        if pos_tup:
            _manage_position(signals, state, symbol, pos_tup, candle, params)
            continue

        cd = GLOBAL["cooldowns"].get(symbol, 0)
        if cd > 0:
            GLOBAL["cooldowns"][symbol] = cd - 1
            continue

        df = state.candles.get((symbol, "1d"))
        if df is None:
            continue
        _maybe_enter(signals, state, symbol, df, candle, params)

    return signals


# ---------------------------------------------------------------------------
# entry / exit helpers
# ---------------------------------------------------------------------------


def start_cooldown(symbol: str, bars: int) -> None:
    GLOBAL["cooldowns"][symbol] = bars


def _maybe_enter(
    signals: list[TradeSignal],
    state: BacktestState,
    symbol: str,
    df: pd.DataFrame,
    candle: Candle,
    params: Params,
) -> None:
    closes = cast(pd.Series, df["close"])
    highs = cast(pd.Series, df["high"])
    lows = cast(pd.Series, df["low"])
    volumes = cast(pd.Series, df["volume"])

    if len(closes) < params.warmup_bars:
        return

    result = detect_cup_and_handle(
        highs,
        lows,
        closes,
        volumes,
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
    entry_price = float(closes.iloc[-1])

    # ATR for risk / trail sizing.
    atr_val = float(ta.atr(highs, lows, closes, params.trail_atr_period).iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = max(handle.cup_depth * 0.1, 1e-9)

    # Fixed per-symbol allocation: size against total starting capital so the
    # symbol deploys ``per_symbol_size`` of the account regardless of cash or
    # of how many other symbols are concurrently positioned.
    qty = per_symbol_qty(
        state.portfolio.initial_capital, params.per_symbol_size, entry_price
    )
    if qty <= 0 or state.portfolio.cash <= 0:
        return

    # Natural stop (structure-based): below the handle low, scaled a touch.
    natural_dist = max(
        params.trail_atr_mult * atr_val, (entry_price - handle.low) * 1.05
    )
    # Risk-budget stop: cap the loss at ``max_risk_pct`` of the account. This
    # is the downside limiter -- it tightens the stop so a stop-out can never
    # lose more than the budgeted fraction.
    risk_cap_dollars = state.portfolio.initial_capital * params.max_risk_pct
    stop_dist = cap_stop_dist(
        natural_dist,
        risk_cap_dollars,
        qty,
        atr_val=atr_val,
        min_stop_atr=params.min_stop_atr,
    )
    stop_loss = round(entry_price - stop_dist, 4)

    # Hard TP applies ONLY when we are NOT in a trending-up regime. In a
    # strong uptrend the fixed target would cap a runner; there we persist no
    # TP and let the (adaptive) ATR trail collect the move.
    trending = is_uptrend(
        highs,
        lows,
        closes,
        adx_window=params.adx_window,
        threshold=params.adx_trend_threshold,
    )
    target = 0.0
    if not trending and params.trend_tp_mult > 0 and handle.cup_depth > 0:
        target = round(entry_price + params.trend_tp_mult * handle.cup_depth, 4)

    GLOBAL["handle_lows"][symbol] = handle.low
    start_cooldown(symbol, params.cooldown_bars)
    signals.append(
        TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            timestamp=candle.timestamp,
            price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=target if target > 0 else None,
            reason=(
                f"[cup&handle] breakout above {handle.breakout_level:.2f} "
                f"(depth={handle.cup_depth:.2f} · {result.reason})"
            ),
        )
    )


def _manage_position(
    signals: list[TradeSignal],
    state: BacktestState,
    symbol: str,
    pos_tup: tuple,
    candle: Candle,
    params: Params,
) -> None:
    df = state.candles.get((symbol, "1d"))
    if df is None:
        return
    closes = cast(pd.Series, df["close"])
    highs = cast(pd.Series, df["high"])
    lows = cast(pd.Series, df["low"])
    close_price = float(closes.iloc[-1])

    handle_low = GLOBAL["handle_lows"].get(symbol)
    atr_val = float(ta.atr(highs, lows, closes, params.trail_atr_period).iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = max(float(closes.iloc[-1]) * 0.01, 1e-9)

    # Adaptive trail: wide in a trend (let the winner run), tight in chop
    # (protect gains); ADX decides the regime.
    trending = is_uptrend(
        highs,
        lows,
        closes,
        adx_window=params.adx_window,
        threshold=params.adx_trend_threshold,
    )
    trail_mult = params.trend_trail_atr_mult if trending else params.chop_trail_atr_mult

    for position in pos_tup:
        # The ratchet is ONLY the exit path for trend entries (no hard TP). For
        # chop entries we persist a fixed TP and let the engine's TP/SL close
        # it -- running a tight ratchet alongside a fixed TP just churns the
        # position into a stack of small exits.
        if position.take_profit is not None:
            continue

        pid = position.position_id
        # RATCHETING trail stop: never retreats. Seeded from the entry stop and
        # handle floor, then ratchets up with ``close - mult*ATR``. A floating
        # trail (recompute from today's close) rides give-backs; a ratchet only
        # moves forward and locks in gains.
        candidate = close_price - trail_mult * atr_val
        floor = handle_low * 0.99 if handle_low else -float("inf")
        seed = float(position.stop_loss) if position.stop_loss else -float("inf")
        base = max(candidate, floor, seed)
        saved = GLOBAL["trail_stops"].get(pid)
        trail_stop = base if saved is None else max(base, saved)
        GLOBAL["trail_stops"][pid] = trail_stop

        if close_price > trail_stop:
            continue

        GLOBAL["trail_stops"].pop(pid, None)
        start_cooldown(symbol, params.cooldown_bars)
        signals.append(
            TradeSignal(
                action=ActionType.close,
                symbol=symbol,
                timestamp=candle.timestamp,
                price=close_price,
                qty=abs(position.qty),
                position_id=position.position_id,
                reason=(
                    f"[cup&handle] ratchet trail hit (close {close_price:.2f} <= {trail_stop:.2f})"
                ),
            )
        )
