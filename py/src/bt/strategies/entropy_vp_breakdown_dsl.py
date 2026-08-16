"""Adaptive-entropy-gated VP breakdown — dynamic ATR trail (DSL).

SHORT-only order-flow breakdown, built on the Volume Profile
(``vp_breakout_dsl``'s short leg) and gated everywhere by the
**adaptive-entropy regime gate**. A short is entered when the per-symbol
adaptive-entropy quantised ``trend`` is at or above the configurable
``min_trend`` floor (``trend >= min_trend``): with ``min_trend=-1`` only
bear regimes trade, with ``min_trend=0`` flat/choppy and bear regimes both
qualify — the strategy fades rallies and breakdowns only, never a confirmed
bull.

Entry signal — a **VP breakdown** (mirrors ``vp_breakout_dsl._enter``'s short
leg, reused here for the same order-flow thesis but gated by entropy instead of
the Kaufman efficiency ratio):

  * ``close < VAL`` — the bar closes below the rolling Value Area Low (close,
    not a wick; ``strength`` near the bar's low),
  * **expanding volume** — the bar's volume ``>= vol_expand_mult *`` the prior
    ``vol_period`` average (order-flow confirmation out of the fair-value
    range),
  * **not in an uptrend** — price on the wrong side of its own rolling mean
    (concurrency filter).

The Value Area comes from an :class:`src.indicators.volume_profile.strategy.
OnlineVP` per symbol in ``ctx.shared``, fed once per candle on the *prior* bars
(no same-bar lookahead — ``OnlineVP.observe`` excludes the current bar).

Exits use a **dynamic ATR stop**: a chandelier-style trailing stop re-anchored
every bar on ``highest(high, trail_lookback) - trail_atr_mult * ATR`` and only
ever moved in our favour (down for a short); it widens with volatility and
tightens against price as the trade works. A revert back **into the value area**
(close > VAL) fires an immediate exit regardless (the institutional tag stop).

All cross-candle state (per-symbol ``OnlineAdaptiveEntropy`` + ``OnlineVP``
instances, cooldowns, per-bar snapshots, dynamic-trail bookkeeping) lives in
``ctx.shared`` via ``@strategy(stateful=True)`` — a fresh holder per run/window,
so there is no cross-window bleed and the strategy is safe across concurrent
split/sweep/optimize workers. Pure helpers take cursor-truncated arrays /
SeriesViews so there is no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.state.types import Position
from src.indicators.volume_profile.strategy import OnlineVP
from src.indicators.adaptive_entropy import AdaptiveEntropyConfig, OnlineAdaptiveEntropy
from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "entropy_vp_breakdown_dsl"

# Keys for per-run state in ctx.shared (minted fresh per run/window).
_VP_KEY = "vp_profiles"  # dict[sym -> OnlineVP]
_SNAP_KEY = "vp_snaps"  # dict[sym -> VolumeProfileSnapshot] (once/bar)
_ENT_KEY = "entropy"  # dict[sym -> OnlineAdaptiveEntropy]
_ENT_RESULT_KEY = "entropy_result"  # dict[sym -> AdaptiveEntropyResult] (once/bar)
_COOLDOWN_KEY = "cooldowns"
_POS_KEY = "positions"  # dict[sym -> dynamic-trail bookkeeping]


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- online volume profile (short leg of vp_breakout_dsl) --
    vp_window: int = 100
    num_bins: int = 50
    value_area_pct: float = 0.70
    vp_warmup: int = 60
    # -- VP breakdown confirmation --
    vol_period: int = 20
    vol_expand_mult: float = 1.3  # breakdown volume >= this x avg
    wick_check: float = 0.5  # close must sit in the lower fraction of the bar
    trend_lookback: int = 50  # concurrency: price vs its own rolling mean
    # -- adaptive-entropy regime gate --
    # Gate threshold on the per-symbol adaptive-entropy quantised ``trend``.
    # A short is allowed when ``trend >= min_trend``: ``-1`` = bear-only
    # (``trend == -1``), ``0`` = non-bull (flat or bear; blocks a confirmed
    # bull). During warmup ``trend`` is 0.
    entropy_lookback: int = 25
    entropy_num_bins: int = 10
    min_trend: int = 0
    # -- dynamic ATR trailing stop --
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    trail_lookback: int = 10
    trail_atr_mult: float = 2.0
    # -- risk / sizing (ATR-risk, 0..1 size) --
    risk_pct: float = 0.01
    # Aggregate |gross notional| cap, fraction of initial capital. Values >= 1.0
    # disable the cap (the ``_enter`` gate requires < 1.0 to fire).
    max_gross_exposure: float = 0.5
    cooldown_bars: int = 5


# ---------------------------------------------------------------------------
# pure helpers (typed; operate on cursor-truncated arrays / SeriesViews only)
# ---------------------------------------------------------------------------


def _last(v: object) -> float:
    """Last visible value of a SeriesView, or a bare float."""
    if isinstance(v, SeriesView):
        return float(v[-1])
    return float(cast(float, v))


def _bar_position_close(close: float, high: float, low: float) -> float:
    """Where the close sits within the bar's range, 0 (low) .. 1 (high)."""
    rng = high - low
    if rng <= 1e-12:
        return 1.0
    return (close - low) / rng


def _avg_volume(ctx: StrategyContext, sym: str, params: Params) -> float:
    """Mean volume over the prior ``vol_period`` bars (excl. the current one)."""
    vol = ctx.ohlcv(sym).volume.to_array()
    n = len(vol)
    if n < params.vol_period + 1:
        return float("nan")
    return float(np.mean(vol[-params.vol_period : -1]))


def _trend_up(ctx: StrategyContext, sym: str, params: Params) -> bool:
    """True when close sits above its own trailing mean (uptrend concurrency)."""
    close = ctx.ohlcv(sym).close.to_array()
    n = len(close)
    if n < params.trend_lookback + 1:
        return False
    ma = float(np.mean(close[-params.trend_lookback :]))
    return float(close[-1]) > ma


def _feed_entropy(ctx: StrategyContext, shared: dict, sym: str, o) -> None:
    """Feed the per-symbol online adaptive-entropy tracker the current bar.

    Called for **every** symbol on **every** bar (top of the per-symbol loop),
    so the tracker warms and its ``trend`` reflects the true ongoing regime —
    not just the sparse bars where a breakdown is being evaluated. The latest
    snapshot is cached for ``bear_gate_ok`` to read synchronously.
    """
    trackers = shared.setdefault(_ENT_KEY, {})
    tr = trackers.get(sym)
    if tr is None:
        cfg = AdaptiveEntropyConfig(
            lookback=ctx.params.entropy_lookback,
            num_bins=ctx.params.entropy_num_bins,
        )
        tr = OnlineAdaptiveEntropy(cfg)
        trackers[sym] = tr
    result = tr.observe(float(o.close[-1]), float(o.high[-1]), float(o.low[-1]))
    shared.setdefault(_ENT_RESULT_KEY, {})[sym] = result


def entropy_result(ctx: StrategyContext, shared: dict, sym: str) -> object:
    """The cached adaptive-entropy snapshot for ``sym`` after feeding this bar."""
    return shared.get(_ENT_RESULT_KEY, {}).get(sym)


def bear_gate_ok(result, min_trend: int) -> bool:
    """Adaptive-entropy regime gate: allow the short above a trend floor.

    A short is permitted when the quantised trend is at or above ``min_trend``
    (e.g. ``min_trend=-1`` = bear-only, ``min_trend=0`` = non-bull). A value of
    ``1`` blocks both flat and bear (effectively off). During warmup the running
    ``trend == 0`` default passes any ``min_trend <= 0``.
    """
    return bool(getattr(result, "trend", 0) >= min_trend)


def _refresh_vp(ctx: StrategyContext, params: Params) -> None:
    """Feed each symbol's online VP once per candle; store the snapshots."""
    profs = ctx.shared.setdefault(_VP_KEY, {})
    snaps: dict[str, object] = {}
    for sym in ctx.symbols:
        prof = profs.get(sym)
        if not isinstance(prof, OnlineVP):
            prof = OnlineVP(
                num_bins=params.num_bins,
                value_area_pct=params.value_area_pct,
                window=params.vp_window,
                warmup_bars=params.vp_warmup,
            )
            profs[sym] = prof
        snaps[sym] = prof.observe(ctx.state, sym, ctx.interval)
    ctx.shared[_SNAP_KEY] = snaps


def _snap(ctx: StrategyContext, sym: str) -> object:
    return ctx.shared.get(_SNAP_KEY, {}).get(sym)


def trail_stop(
    highs: np.ndarray, atr: float, trail_lookback: int, mult: float
) -> float:
    """Chandelier ATR trail for a short: ``highestHigh - mult * ATR``.

    Anchors on the highest high of the most recent ``trail_lookback`` bars
    (cursor-truncated ``highs``) minus ``mult * ATR``. Returns NaN when ATR is
    non-finite/non-positive or no highs are visible.
    """
    if not np.isfinite(atr) or atr <= 0 or len(highs) < 1:
        return float("nan")
    lo = max(0, len(highs) - trail_lookback)
    anchor = float(highs[lo:].max())
    return anchor - mult * atr


def _size(ctx: StrategyContext, params: Params, price: float, atr_val: float) -> float:
    """0..1 capital fraction to deploy, ATR-risk sized on current cash."""
    if np.isnan(atr_val) or atr_val <= 0 or price <= 0:
        return 0.0
    cash = ctx.state.portfolio.cash
    if cash <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=cash,
        price=price,
        stop_dist=params.stop_atr_mult * atr_val,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_COOLDOWN_KEY, {})
    shared.setdefault(_POS_KEY, {})

    _refresh_vp(ctx, params)

    # ---- Advances / exits first --------------------------------------------
    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        _feed_entropy(ctx, shared, sym, o)  # regime tracker every bar

        pos = ctx.position(sym)
        if pos is None:
            continue
        _manage_open(ctx, shared, params, sym, o, pos)

    # ---- Entries ------------------------------------------------------------
    for sym in ctx.symbols:
        cd = shared[_COOLDOWN_KEY].get(sym, 0)
        if cd > 0:
            shared[_COOLDOWN_KEY][sym] = cd - 1
            continue
        if ctx.position(sym) is not None:
            continue
        _enter(ctx, sym, params)


# ── entry (VP breakdown gated by bear regime) ────────────────────────────────


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    snap = _snap(ctx, sym)
    if snap is None or not getattr(snap, "ready", False):
        return
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < max(params.vp_warmup, params.trend_lookback + 1, params.vol_period + 1):
        return

    close = float(o.close[-1])
    high = float(o.high[-1])
    low = float(o.low[-1])
    val = getattr(snap, "val", None)
    if not isinstance(val, float):
        return

    atr_val = _last(ctx.ta.atr(sym, params.atr_period))
    if close <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return

    # Adaptive-entropy regime gate: never fade a confirmed bull (or tighter).
    ent = entropy_result(ctx, ctx.shared, sym)
    if not bear_gate_ok(ent, params.min_trend):
        return

    avg_vol = _avg_volume(ctx, sym, params)
    if np.isnan(avg_vol) or avg_vol <= 0:
        return
    vol_expanding = float(o.volume[-1]) >= params.vol_expand_mult * avg_vol
    strength = _bar_position_close(close, high, low)

    short_breakdown = (
        close < val  # close (not a wick) breaks the value area top
        and strength <= 1.0 - params.wick_check  # held near the lows
        and vol_expanding  # order-flow confirmation
        and not _trend_up(ctx, sym, params)  # concurrency: not against trend
    )
    if not short_breakdown:
        return

    size = _size(ctx, params, close, atr_val)
    if size <= 0:
        return

    # Aggregate gross-exposure cap: skip if this entry would push total |gross|
    # notional past ``max_gross_exposure`` of initial capital. A value >= 1.0
    # disables the cap (100%+ of capital is effectively unconstrained for a
    # risk-sized book).
    if 0.0 < params.max_gross_exposure < 1.0:
        from src.bt.strategies.vp_breakout_dsl import _gross_exposure

        if (
            _gross_exposure(
                ctx.state.portfolio.initial_capital,
                ctx.state.portfolio.positions,
            )
            + size
            >= params.max_gross_exposure
        ):
            return

    # Initial stop for sizing is one ATR distance above the entry; the dynamic
    # trailing stop is established from here and ratcheted down each bar.
    stop_price = close + params.stop_atr_mult * atr_val
    ctx.short(
        sym,
        size=size,
        reason=f"[breakdown] short close {close:.2f} < VAL {val:.2f} in bear regime",
    )
    ctx.shared[_POS_KEY][sym] = {"entry": close, "trail": stop_price, "pid": ""}


# ── position management (dynamic ATR trailing stop) ──────────────────────────


def _manage_open(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
    pos: Position,
) -> None:
    """Manage an open short: value-area revert + dynamic ATR trailing stop.

    The trail is re-anchored every bar on the highest high of the last
    ``trail_lookback`` bars minus ``trail_atr_mult * ATR``, and only ever moved
    *down* for a short (a profitable position tightens the stop; a rally keeps
    the best previous protection). Exits when close touches the trail or price
    reverts back into the value area.
    """
    rec = shared[_POS_KEY].get(sym)
    atr = _last(ctx.ta.atr(sym, params.atr_period))
    close = float(o.close[-1])

    if rec is not None and not rec.get("pid"):
        rec["pid"] = pos.position_id

    # (1) Value-area revert (institutional tag stop): close back inside the
    # fair-value range invalidates the breakdown.
    snap = _snap(ctx, sym)
    val = getattr(snap, "val", None) if snap is not None else None
    if isinstance(val, float) and close > val:
        _exit(ctx, shared, params, sym, "reverted into value area")
        return

    # (2) Dynamic ATR trailing stop, only ever ratcheted down.
    if rec is not None and np.isfinite(atr):
        new_trail = trail_stop(
            o.high.to_array(), atr, params.trail_lookback, params.trail_atr_mult
        )
        if np.isfinite(new_trail) and new_trail < rec["trail"]:
            rec["trail"] = new_trail
        if close >= rec["trail"]:
            _exit(ctx, shared, params, sym, "dynamic ATR trail hit")
            return


def _exit(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    reason: str,
) -> None:
    shared[_POS_KEY].pop(sym, None)
    shared[_COOLDOWN_KEY][sym] = params.cooldown_bars
    ctx.close(sym, reason=f"[breakdown] {reason}")
