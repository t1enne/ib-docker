"""Biotech hourly failed-down-breakout long (DSL) — iteration-2 candidate.

LONG the structural *failed down-breakout* on a venue volume spike, then hard-
stop on ATR and take profit into the pre-spike mean. This is the sign-mirror of
``bio_post_catalyst_fade_dsl`` (iteration-1, a SHORT of the failed *up*-breakout
that measured net -0.87% / Sharpe -0.165) — SAME touch-and-reject + volume-spike
entry geometry, opposite side, on the same biotech 1h universe (XBI bench).

Iteration-1 transferable lesson being tested: up-side catalyst spikes *continued*
(its shorts bled into adverse overnight gaps — first MRNA short gapped ~-$995
through the 1.5xATR stop). The hypothesis here is asymmetric: biotechs gap DOWN
on neutral/washout news less violently than they rip UP on good news, so the
DOWN-side fake-breakout reversion may carry the OPPOSITE (positive-to-a-long)
skew vs the up-side fade that bled.

Entry — a *failed down-breakout*, volume-first (no price-only dip-buy):
  * the current bar's LOW pokes at least ``extend_pct`` below the rolling
    ``range_bars``-lookback prior-shelf LOW (measured before the current bar —
    no lookahead),
  * the current bar CLOSES back at/above that shelf (touch-and-reject, not a
    sustained breakdown),
  * the current bar's volume is >= ``vol_mult *`` the mean of the prior
    ``vol_period`` bars (the single direct volume trigger — squarely off the
    noise floor; no signal fires without it, a flat/absent read kills the long
    outright),
  * a per-symbol cooldown since the last exit has elapsed (catalyst moves
    cluster),
  * optionally require the pre-spike base not already in free-fall
    (``require_flat_base``): the reference mean must not be plunging hard, so we
    do not buy a name already mid-collapse (a fresh down-leg often arrives).

Exits (planned before entry; every exit through ``ctx.close`` fills at the next
bar's open, so an overnight gap adverse to a long — down through entry — is
realized at the open, never never-realized hope):
  * invalidation: the close LEAKS back below the just-rejected low shelf minus
    ``shelf_tol * ATR`` — the "breakdown actually held" tag stop (the mirror of
    iteration-1's shelf-reclaim),
  * ATR stop: ``close <= entry - stop_atr_mult * ATR`` — hard, sized off ATR,
  * target: drift back up into / above the pre-spike mean (``target_atr_rng``) ->
    take the reversion profit mechanically.

Gap-risk posture (mirror of iteration-1, re-stated for a long): the adverse
vector is now a DOWN-gap through a long. position size is derived from a tight
``risk_pct`` against the ATR stop distance, and a stop does not protect a long
someone gaps through down overnight — that risk is priced into the risk budget
and surfaced in the run's honest risk sheet, not hidden by a wider stop.

All cross-candle state (per-symbol cooldowns, open-trade structural bookkeeping)
lives in ``ctx.shared`` via ``@strategy(stateful=True)`` — a fresh holder per
run/window, safe across concurrent split/sweep/optimize workers. Pure helpers
operate only on cursor-truncated SeriesViews/numpy arrays (no lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams
from src.bt.size.pure import risk_sized_qty

STRATEGY_TYPE = "bio_failed_down_breakout_long_dsl"

# ctx.shared keys (module-private; minted fresh per run/window).
_CD_KEY = "cd_key"  # dict[sym -> int] remaining entry cooldown bars
_OPEN_KEY = "open_trades"  # dict[sym -> dict] structural levels for an open long


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- volume-spike confirmation (the direct volume trigger) --
    vol_period: int = 24  # prior bars over which volume is averaged
    vol_mult: float = 3.0  # entry-bar volume >= this * mean(prior vol_period)
    # -- structural failed-breakdown geometry (long mirror) --
    range_bars: int = 48  # lookback for the prior-shelf low (before current bar)
    extend_pct: float = 0.005  # low must poke this fraction below prior shelf low
    require_flat_base: bool = True  # do not buy a base already collapsing hard
    flat_base_slope_atr: float = 0.5  # base-mean drop threshold to stay flat
    # -- stops / target / risk --
    atr_period: int = 14
    stop_atr_mult: float = 1.5  # hard ATR stop below entry
    shelf_tol_atr: float = 0.15  # invalidation tolerance below the rejected shelf
    risk_pct: float = 0.005  # 0.5% equity risked per trade (gap tail is real)
    target_atr_rng: float = 1.0  # take profit once price climbs >= this ATR above entry
    cooldown_bars: int = 96  # hours to leave a symbol alone after a long exits
    # -- lifecycle --
    min_base_bars: int = 40  # minimum prior bars before a buy is even considered


# ---------------------------------------------------------------------------
# pure helpers (typed; operate on cursor-truncated Series/arrays only)
# ---------------------------------------------------------------------------


def _last_array(v: SeriesView) -> float:
    """Current (cursor) scalar of a SeriesView."""
    return float(v[-1])


def _visible(a: np.ndarray) -> np.ndarray:
    """Cursor-truncated copy, NaN-filtered (mirrors TaContext feed alignment)."""
    return np.asarray(a, dtype=float)[np.isfinite(a)]


def prior_shelf_low(lows: SeriesView, range_bars: int) -> float:
    """Minimum low over the ``range_bars`` bars immediately before the current.

    Excludes the current bar (the breakdown spike bar) so the sensor measures a
    *prior* established demand shelf and can detect a poke *below* it, never
    against itself. NaN when not enough prior bars are visible.
    """
    arr = _visible(lows.to_array())
    if len(arr) < range_bars + 1:
        return float("nan")
    prior = arr[-(range_bars + 1) : -1]
    return float(np.min(prior))


def mean_prior_volume(volume: SeriesView, vol_period: int) -> float:
    """Mean volume over the ``vol_period`` bars before the current (excl. it)."""
    arr = _visible(volume.to_array())
    if len(arr) < vol_period + 1:
        return float("nan")
    return float(np.mean(arr[-(vol_period + 1) : -1]))


def mean_prior_close(closes: SeriesView, window: int) -> float:
    """Mean close over the ``window`` bars before the current (excl. it)."""
    arr = _visible(closes.to_array())
    if len(arr) < window + 1:
        return float("nan")
    return float(np.mean(arr[-(window + 1) : -1]))


def base_not_plunging(
    closes: SeriesView, window: int, flat_base_slope_atr: float, atr_val: float
) -> bool:
    """True when the pre-spike base is not itself collapsing hard.

    Long-mirror of iteration-1's ``base_is_flat_enough``
    (which rejected a base climbing hard for the short here). Measured as the
    newest vs oldest half of the ``window`` prior-base closes (both strictly
    prior to the current spike bar). If the newer half mean *falls* below the
    older half mean by more than ``flat_base_slope_atr * ATR`` the base is a
    collapse-in-progress and we decline the buy (a fresh catalyst down-leg is
    likely ahead).
    """
    a = _visible(closes.to_array())
    half = max(1, window // 2)
    if len(a) < half * 2 + 1 or not np.isfinite(atr_val) or atr_val <= 0:
        return True  # cannot measure a collapse -> do not block (safe default)
    older_half = a[-(half * 2 + 1) : -(half + 1)]
    newer_half = a[-(half + 1) : -1]
    drift = float(np.mean(newer_half) - np.mean(older_half))
    return drift >= -flat_base_slope_atr * atr_val


def attr_avg_true_range(ctx: StrategyContext, sym: str, period: int) -> float:
    """Latest ATR scalar for ``sym``, or NaN when not yet computable."""
    try:
        v = ctx.ta.atr(sym, period)
    except KeyError:
        return float("nan")
    if len(v) == 0:
        return float("nan")
    return float(v[-1])


def size_frac(
    ctx: StrategyContext,
    price: float,
    atr_val: float,
    stop_mult: float,
    risk_pct: float,
) -> float:
    """0..1 capital fraction, ATR-risk sized off current cash (risk/stop)."""
    if price <= 0 or not np.isfinite(atr_val) or atr_val <= 0 or risk_pct <= 0:
        return 0.0
    cash = ctx.state.portfolio.cash
    if cash <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=cash,
        price=price,
        stop_dist=stop_mult * atr_val,
        risk_pct=risk_pct,
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return float(min(max(size, 0.0), 1.0))


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    shared = ctx.shared
    params: Params = ctx.params
    shared.setdefault(_CD_KEY, {})
    shared.setdefault(_OPEN_KEY, {})

    # decrement cooldowns first so an open buy never counts itself as fresh
    _tick_cooldowns(shared)

    # manage any open long exits before we look for new entries
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            _manage_open(ctx, sym, params)

    # entries (volume-first, structural failed-down-breakout long)
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            continue
        if shared[_CD_KEY].get(sym, 0) > 0:
            continue
        _enter(ctx, sym, params)


def _tick_cooldowns(shared: dict) -> None:
    cd = shared[_CD_KEY]
    for sym, left in cd.items():
        if left > 0:
            cd[sym] = left - 1


# ── entry: failed down-breakout on a volume spike (long mirror) ─────────────


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    o = ctx.ohlcv(sym)
    n = len(o.close)
    need = params.min_base_bars + 1
    if n < need:
        return

    atr_val = attr_avg_true_range(ctx, sym, params.atr_period)
    if not np.isfinite(atr_val) or atr_val <= 0:
        return

    low = _last_array(o.low)
    close = _last_array(o.close)
    shelf = prior_shelf_low(o.low, params.range_bars)
    base_mean = mean_prior_close(o.close, params.range_bars)
    if not np.isfinite(shelf) or not np.isfinite(base_mean) or base_mean <= 0:
        return

    # VOLUME FIRST — the direct volume read is the single entry trigger.
    avg_vol = mean_prior_volume(o.volume, params.vol_period)
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        return
    if float(o.volume[-1]) < params.vol_mult * avg_vol:
        return  # no volume spike -> no long, regardless of price geometry

    # Wait for a visibly extended shelf poke (below) and a close BACK above it.
    if low >= shelf * (1.0 - params.extend_pct):
        return  # never poked meaningfully below the prior shelf low
    if close < shelf:
        return  # leaked below the shelf on the close -> not a failed washout

    # Structural context: do not buy a base already collapsing hard.
    if params.require_flat_base and not base_not_plunging(
        o.close, params.range_bars, params.flat_base_slope_atr, atr_val
    ):
        return

    # A buy only makes sense while price is still below the pre-spike mean; a
    # close already back above it is not "washed out below" any more.
    if close >= base_mean:
        return

    size = size_frac(ctx, close, atr_val, params.stop_atr_mult, params.risk_pct)
    if size <= 0:
        return

    stop_dist = params.stop_atr_mult * atr_val
    stop_price = close - stop_dist
    if low <= stop_price:
        return  # stop inside the breakdown range -> skip rather than size to a gap

    ctx.long(
        sym,
        size=size,
        reason=(
            f"[bio-long] long failed down-breakout close {close:.2f} "
            f"vs shelf {shelf:.2f} on vol {params.vol_mult}x spike"
        ),
    )
    shared = ctx.shared
    shared[_OPEN_KEY][sym] = {
        "entry": close,
        "shelf": shelf,
        "stop": stop_price,
        "entry_atr": atr_val,
    }


# ── position management ─────────────────────────────────────────────────────


def _manage_open(ctx: StrategyContext, sym: str, params: Params) -> None:
    """Manage an open long: ATR stop, breakdown-confirm invalidation, reversion TP.

    Called only for symbols holding a position. Exits fire via ``ctx.close``
    (fills at the next bar's open) so an overnight gap adverse to the long
    (down through entry) is realized at the gap open rather than hope-held.
    """
    shared = ctx.shared
    rec = shared.get(_OPEN_KEY, {}).get(sym)
    if rec is None:
        return  # not our long (e.g. inherited engine position) -> leave alone
    o = ctx.ohlcv(sym)
    close = _last_array(o.close)
    atr_val = attr_avg_true_range(ctx, sym, params.atr_period)
    if not np.isfinite(atr_val) or atr_val <= 0:
        # can't assess risk cleanly -> bail out of an unmanageable hold
        _flatten(ctx, shared, sym, "atr unavailable (exit open long)")
        return

    entry = float(rec["entry"])
    shelf = float(rec["shelf"]) if np.isfinite(rec["shelf"]) else float("nan")
    # stop can only tighten for a long (roll to a higher lower-of level), never
    # loosen into the pure gap-loss direction.
    stop_level = max(float(rec["stop"]), entry - params.stop_atr_mult * atr_val)

    if close >= entry + params.target_atr_rng * atr_val:
        _flatten(ctx, shared, sym, "reversion target reached (take profit)")
    elif np.isfinite(shelf) and close <= shelf - params.shelf_tol_atr * atr_val:
        _flatten(ctx, shared, sym, "shelf no longer held (breakdown confirmed)")
    elif close <= stop_level:
        _flatten(ctx, shared, sym, "ATR stop hit")
    else:
        rec["stop"] = stop_level  # tighten but keep holding
        shared[_OPEN_KEY][sym] = rec


def _flatten(ctx: StrategyContext, shared: dict, sym: str, reason: str) -> None:
    """Remove bookkeeping and close every open lot in ``sym``."""
    shared[_OPEN_KEY].pop(sym, None)
    cd = shared.get(_CD_KEY, {})
    if sym in cd:
        cd[sym] = int(ctx.params.cooldown_bars)
    ctx.close(sym, reason=f"[bio-long] {reason}")
