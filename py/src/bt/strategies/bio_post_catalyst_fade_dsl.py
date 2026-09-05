"""Biotech hourly post-catalyst fade (DSL) — iteration-1 candidate.

SHORT the structural *failed up-breakout* on a venue volume spike, then hard-
stop on ATR and take profit into the pre-spike mean. Built specifically for the
biotech-universe catalyst profile (XBI bench) on 1h bars, where a one-bar rip
that pokes above a prior measured shelf and closes back inside/below it on
expanding volume is a short-rich trap.

Entry — a *failed up-breakout*, volume-first (no price-only fade):
  * the current bar's high pokes at least ``extend_pct`` above the rolling
    ``range_bars``-lookback prior-shelf high (measured before the current bar),
  * the current bar CLOSES back at/below that shelf (touch-and-reject, not a
    sustained breakout),
  * the current bar's volume is >= ``vol_mult *`` the mean of the prior
    ``vol_period`` bars (the single direct volume trigger — squarely off the
    noise floor; no signal fires without it),
  * a per-symbol cooldown bars since the last exit has elapsed (catalyst
    moves cluster; do not re-short a name already fading out of sync).
  * Optionally require the pre-spike base to be quiet (see ``require_flat_base``):
    the reference mean must not itself be climbing hard, so we don't fade a
    name that is already mid-extended-run (where a fresh leg often arrives).

Exits (planned before entry; every one through ``ctx.close`` — no gap-sniped
hope-holds):
  * invalidation: the close reclaims the just-rejected shelf + ``shelf_tol``
    ``*`` ATR — the "breakout actually held" tag stop that also gates the gap
    vector (a next-bar gap above the shelf kills the short regardless),
  * ATR stop: ``close >= entry + stop_atr_mult * ATR`` — hard, sized off ATR,
  * target: drift back into / below the pre-spike mean (``entry_atr_rng``) ->
    take the reversion profit mechanically.

Volume is the ONLY entry trigger (per the trader's volume-first rule). Price
distance alone never shorts. The direct volume read (current bar vs its own
prior mean) must be clearly off the floor for any gate to pass.

Gap-risk posture (measured empirically: names like MRNA/SRPT can gap >10% on
single catalyst days, worst 1h-range ~13-27%): position size is derived from a
tight ``risk_pct`` against the ATR stop distance, and a stop *does not protect* a
short that someone else gaps through overnight — that risk is priced into the
risk budget and explicitly surfaced in the run's honest risk sheet, not hidden
by a wider stop.

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

STRATEGY_TYPE = "bio_post_catalyst_fade_dsl"

# ctx.shared keys (module-private; minted fresh per run/window).
_CD_KEY = "cd_key"  # dict[sym -> int] remaining entry cooldown bars
_OPEN_KEY = "open_trades"  # dict[sym -> dict] structural levels for an open short


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- volume-spike confirmation (the direct volume trigger) --
    vol_period: int = 24  # prior bars over which volume is averaged
    vol_mult: float = 3.0  # entry-bar volume >= this * mean(prior vol_period)
    # -- structural failed-breakout geometry --
    range_bars: int = 48  # lookback for the prior-shelf high (before current bar)
    extend_pct: float = 0.005  # high must poke this fraction above prior shelf
    require_flat_base: bool = True  # do not fade a base already climbing hard
    flat_base_slope_atr: float = 0.5  # base mean ATR-move threshold to stay flat
    # -- stops / target / risk --
    atr_period: int = 14
    stop_atr_mult: float = 1.5  # hard ATR stop above entry
    shelf_tol_atr: float = 0.15  # invalidation tolerance above the rejected shelf
    risk_pct: float = 0.005  # 0.5% equity risked per trade (gap tail is real)
    target_atr_rng: float = 1.0  # take profit once price falls >= this ATR below entry
    cooldown_bars: int = 96  # hours to leave a symbol alone after a fade exits
    # -- lifecycle --
    min_base_bars: int = 40  # minimum prior bars before a fade is even considered


# ---------------------------------------------------------------------------
# pure helpers (typed; operate on cursor-truncated Series/arrays only)
# ---------------------------------------------------------------------------


def _last_array(v: SeriesView) -> float:
    """Current (cursor) scalar of a SeriesView."""
    return float(v[-1])


def _visible(a: np.ndarray) -> np.ndarray:
    """Cursor-truncated copy, NaN-filtered (mirrors TaContext feed alignment)."""
    return np.asarray(a, dtype=float)[np.isfinite(a)]


def prior_shelf_high(highs: SeriesView, range_bars: int) -> float:
    """Maximum high over the ``range_bars`` bars immediately before the current.

    Excludes the current bar (the spike bar) so the sensor measures a *prior*
    established shelf and can detect a poke *above* it, never against itself.
    Returns NaN when not enough prior bars are visible.
    """
    arr = _visible(highs.to_array())
    if len(arr) < range_bars + 1:
        return float("nan")
    prior = arr[-(range_bars + 1) : -1]
    return float(np.max(prior))


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


def base_is_flat_enough(
    closes: SeriesView, window: int, flat_base_slope_atr: float, atr_val: float
) -> bool:
    """True when the pre-spike base mean is not itself climbing hard.

    Measured as the oldest vs newest half of the ``window`` prior-base means
    (both strictly prior to the current spike bar). If the difference exceeds
    ``flat_base_slope_atr * ATR`` the base is an extended run-in-progress and
    we decline the fade (a fresh catalyst leg is likely ahead).
    """
    a = _visible(closes.to_array())
    half = max(1, window // 2)
    if len(a) < half * 2 + 1 or not np.isfinite(atr_val) or atr_val <= 0:
        return False
    older_half = a[-(half * 2 + 1) : -(half + 1)]
    newer_half = a[-(half + 1) : -1]
    drift = float(np.mean(newer_half) - np.mean(older_half))
    return drift <= flat_base_slope_atr * atr_val


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

    # decrement cooldowns first so an open fade never counts itself as fresh
    _tick_cooldowns(shared)

    # manage any open short exits before we look for new entries
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            _manage_open(ctx, sym, params)

    # entries (volume-first, structural failed-up-breakout)
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


# ── entry: failed up-breakout on a volume spike ─────────────────────────────


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    o = ctx.ohlcv(sym)
    n = len(o.close)
    need = params.min_base_bars + 1
    if n < need:
        return

    atr_val = attr_avg_true_range(ctx, sym, params.atr_period)
    if not np.isfinite(atr_val) or atr_val <= 0:
        return

    high = _last_array(o.high)
    close = _last_array(o.close)
    shelf = prior_shelf_high(o.high, params.range_bars)
    base_mean = mean_prior_close(o.close, params.range_bars)
    if not np.isfinite(shelf) or not np.isfinite(base_mean) or base_mean <= 0:
        return

    # VOLUME FIRST — the direct volume read is the single entry trigger.
    avg_vol = mean_prior_volume(o.volume, params.vol_period)
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        return
    if float(o.volume[-1]) < params.vol_mult * avg_vol:
        return  # no volume spike -> no fade, regardless of price geometry

    # Wait for a visibly extended shelf poke and a close BACK inside it.
    if high <= shelf * (1.0 + params.extend_pct):
        return  # never poked meaningfully above the prior shelf
    if close > shelf:
        return  # held the breakout on the close -> not a failed pop

    # Structural context: reject a base already rising hard (recent extended leg).
    if params.require_flat_base and not base_is_flat_enough(
        o.close, params.range_bars, params.flat_base_slope_atr, atr_val
    ):
        return

    # A fade only makes sense while price is still above the pre-spike mean;
    # a close already back under it is not "extended above" any more.
    if close <= base_mean:
        return

    size = size_frac(ctx, close, atr_val, params.stop_atr_mult, params.risk_pct)
    if size <= 0:
        return

    stop_dist = params.stop_atr_mult * atr_val
    stop_price = close + stop_dist
    if high >= stop_price:
        return  # stop inside the spike range -> skip rather than size to a gap

    ctx.short(
        sym,
        size=size,
        reason=(
            f"[bio-fade] short failed up-breakout close {close:.2f} "
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
    """Manage an open short: ATR stop, shelf-reclaim invalidation, reversion TP.

    Called only for symbols holding a position. Exits fire via ``ctx.close``
    (fills at the next bar's open) so an overnight gap adverse to the short is
    realized at the gap open rather than never realized.
    """
    shared = ctx.shared
    rec = shared.get(_OPEN_KEY, {}).get(sym)
    if rec is None:
        return  # not our short (e.g. inherited engine position) -> leave alone
    o = ctx.ohlcv(sym)
    close = _last_array(o.close)
    atr_val = attr_avg_true_range(ctx, sym, params.atr_period)
    if not np.isfinite(atr_val) or atr_val <= 0:
        # can't assess risk cleanly -> bail out of an unmanageable hold
        _flatten(ctx, shared, sym, "atr unavailable (exit open short)")
        return

    entry = float(rec["entry"])
    shelf = float(rec["shelf"]) if np.isfinite(rec["shelf"]) else float("nan")
    # stop can only tighten for a short (roll to a lower/higher-of level), never
    # loosen into the pure gap-loss direction.
    stop_level = min(float(rec["stop"]), entry + params.stop_atr_mult * atr_val)

    if close <= entry - params.target_atr_rng * atr_val:
        _flatten(ctx, shared, sym, "reversion target reached (take profit)")
    elif np.isfinite(shelf) and close >= shelf + params.shelf_tol_atr * atr_val:
        _flatten(ctx, shared, sym, "shelf reclaimed (breakout held -> invalidated)")
    elif close >= stop_level:
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
    ctx.close(sym, reason=f"[bio-fade] {reason}")
