"""Price-RSI bullish divergence reversal — signal-then-confirm (DSL).

LONG-only countertrend-to-shallow-correction reversion, quality-only setups. A
genuine bullish price-RSI divergence (recent swing low makes a *lower* price low
while RSI makes a *higher* low and sits genuinely oversold) is stored as a
**pending signal**; it is *not* entered on the divergence bar. The optional
``use_trend_gate`` parameter toggles the slow-EMA filter: when ``True`` the
signal is additionally gated by a bullish EMA trend (``fast_ema > slow_ema``);
when ``False`` (the default) the trade is taken on the divergence's own terms —
pure oversold divergence reversion, which fires far more often and is what
catches reversals in trend breakouts/bear legs. Either way the position is
*not* entered on the divergence bar. On a later bar (within ``max_confirm_latency``) that
satisfies the confirmation gates — close back above the fast EMA, RSI not
vertical-ripped, rising volume participation, trend still up — the position is
entered with an ATR-risk-sized stop. A divergence that fails to confirm within
the latency window is dropped as stale. Positions are managed: swing-low
invalidation, RSI-overbought close, and a take-profit partial.

The signal/confirmation split is deliberate: a fresh oversold divergence prints
a price *low* today, which sits below the fast EMA by definition, so demanding
``close > EMA20`` on the same bar is structurally impossible. Waiting for a
confirmation bar that closes back above the EMA avoids catching the falling
knife and is what makes the edge tradable.

All cross-candle state (cooldowns, per-symbol pending signals, open-position
bookkeeping) lives in ``ctx.shared`` via ``@strategy(stateful=True)`` — the DSL
per-run holder, never module globals. Pure helpers take cursor-truncated numpy
arrays, so there is no lookahead and the helpers are directly unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.size.pure import equity_of, risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "rsi_divergence_dsl"

#: ATR period for stop/target risk (no dedicated param; matches ta default).
_ATR_PERIOD = 14
#: Close the position when RSI exceeds this (overbought -> take the gift).
_OVERBOUGHT = 70.0
#: Fraction of the open lot shed at the ATR take-profit target.
_PARTIAL_QTY = 0.5


@dataclass(frozen=True)
class Params(StrategyParams):
    rsi_period: int = 14
    pivot_lookback: int = 5
    fast_ma: int = 20
    slow_ma: int = 63
    confirm_vol_window: int = 5
    stop_atr_mult: float = 2.0
    risk_pct: float = 0.01
    take_profit_atr: float = 3.0
    cooldown_bars: int = 10
    min_lookback: int = 100
    rsi_floor: float = 35.0
    max_rsi_at_trigger: float = 60.0
    max_confirm_latency: int = 5
    #: Optional slow-EMA trend filter on the entry. Default False -> trade the
    #: pure oversold divergence (higher trade count); True -> trend-confirmed.
    use_trend_gate: bool = False


# ---------------------------------------------------------------------------
# pure helpers (fully typed; operate on cursor-truncated numpy arrays)
# ---------------------------------------------------------------------------


def find_swing_lows(lows: np.ndarray, lookback: int) -> list[int]:
    """Indices of fractal swing lows: bar ``i`` is strictly below the
    ``lookback`` lows on each side."""
    n = len(lows)
    out: list[int] = []
    for i in range(lookback, n - lookback):
        left = lows[i - lookback : i]
        right = lows[i + 1 : i + lookback + 1]
        if lows[i] < left.min() and lows[i] < right.min():
            out.append(i)
    return out


def detect_bullish_divergence(
    lows: np.ndarray,
    rsi: np.ndarray,
    lookback: int,
    rsi_floor: float,
) -> tuple[int, int] | None:
    """Return ``(cur_low_i, prev_low_i)`` of a fresh bullish price-RSI
    divergence, else ``None``.

    The most recent swing low makes a *lower* price low than the previous swing
    low while its RSI is a *higher* low and sits below ``rsi_floor`` (genuinely
    oversold). The swing low must be recent (within ``lookback*2`` bars of the
    current bar) so the divergence is fresh, not stale. No lookahead — ``lows``/
    ``rsi`` are already cursor-truncated.
    """
    n = len(lows)
    if n < 2 * lookback + 1:
        return None
    idxs = find_swing_lows(lows, lookback)
    if len(idxs) < 2:
        return None
    cur_i, prev_i = idxs[-1], idxs[-2]
    if n - 1 - cur_i > lookback * 2:
        return None
    cur_rsi, prev_rsi = rsi[cur_i], rsi[prev_i]
    if not (np.isfinite(cur_rsi) and np.isfinite(prev_rsi)):
        return None
    if not (lows[cur_i] < lows[prev_i]):  # price lower low
        return None
    if not (prev_rsi < cur_rsi):  # RSI higher low -> divergence
        return None
    if not (cur_rsi < rsi_floor):
        return None
    return (cur_i, prev_i)


def trigger_confirmation(vol: np.ndarray, confirm_vol_window: int) -> bool:
    """Rising-participation confirmation for the current (confirmation) bar.

    The current bar's volume must be ``>=`` the rolling mean of the last
    ``confirm_vol_window`` bars (inclusive of the current bar) *and* ``>=`` the
    previous bar's volume — "buyers stepped in and stayed". Returns False on
    too-short series or a non-positive window mean.
    """
    n = len(vol)
    if n < 2:
        return False
    cur = vol[-1]
    window_start = max(0, n - confirm_vol_window)
    window_mean = float(vol[window_start:].mean())
    if not (window_mean > 0):
        return False
    return bool(cur >= window_mean and cur >= vol[-2])


def trend_bullish(fast_ema: float, slow_ema: float) -> bool:
    """True when the fast EMA is above the slow EMA (both finite)."""
    if not (np.isfinite(fast_ema) and np.isfinite(slow_ema)):
        return False
    return fast_ema > slow_ema


def stop_distance(
    entry: float, swing_low: float, atr: float, stop_atr_mult: float
) -> float:
    """Risk distance from entry to the stop level, floored at one ATR.

    The stop sits ``stop_atr_mult * ATR`` below the swing low; the distance
    from ``entry`` to that stop is ``entry - stop_price``, never allowed below
    one ATR so the stop cannot be too tight for the volatility.
    """
    stop_price = swing_low - stop_atr_mult * atr
    return max(atr, entry - stop_price)


def entry_size(qty: float, price: float, initial_capital: float) -> float:
    """Back-solve the ``size`` (0..1 fraction of initial capital) for a
    risk-targeted absolute ``qty`` (inverse of the DSL's fixed-size emission)."""
    if price <= 0 or initial_capital <= 0:
        return 0.0
    return qty * price / initial_capital


def _last(v: object) -> float:
    """Last visible value of a SeriesView, or a bare float."""
    if isinstance(v, SeriesView):
        return float(v[-1])
    return float(cast(float, v))


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    shared = ctx.shared
    shared.setdefault("cooldowns", {})
    shared.setdefault("positions", {})
    shared.setdefault("pending", {})
    params = ctx.params

    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        if len(o.close) < params.min_lookback:
            continue

        if ctx.position(sym) is not None:
            # An open position supersedes any pending signal for this symbol.
            shared["pending"].pop(sym, None)
            _manage_open(ctx, shared, params, sym, o)
            continue

        cd = shared["cooldowns"].get(sym, 0)
        if cd > 0:
            # Cooldown blocks both fresh detection AND pending confirmation.
            shared["pending"].pop(sym, None)
            shared["cooldowns"][sym] = cd - 1
            continue

        if sym in shared["pending"]:
            if _try_confirm(ctx, shared, params, sym, o):
                shared["pending"].pop(sym, None)
            else:
                _age_pending(shared, params, sym)
            continue

        _try_store_signal(ctx, shared, params, sym, o)


def _manage_open(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
) -> None:
    """Exit/hike management for an open position (never average down)."""
    rec = shared["positions"].get(sym)
    atr = _last(ctx.ta.atr(sym, _ATR_PERIOD))
    rsi = _last(ctx.ta.rsi(sym, params.rsi_period))
    close = float(o.close[-1])
    # Capture the lot handle once the fill lands (pid unknown at emit time).
    if rec is not None and not rec.get("pid"):
        p = ctx.position(sym)
        if p is not None:
            rec["pid"] = p.position_id

    if rec is not None and close <= rec["swing_low"]:
        _exit(ctx, shared, sym, "swing low invalidation")
        return
    if np.isfinite(rsi) and rsi > _OVERBOUGHT:
        _exit(ctx, shared, sym, "rsi overbought")
        return
    if (
        rec is not None
        and np.isfinite(atr)
        and close >= rec["entry"] + params.take_profit_atr * atr
    ):
        ctx.partial_close(
            sym,
            qty=_PARTIAL_QTY,
            lot=rec.get("pid") or "",
            reason="[rsi-div] partial @ target",
        )


def _exit(ctx: StrategyContext, shared: dict, sym: str, reason: str) -> None:
    """Full close + per-symbol cooldown."""
    shared["positions"].pop(sym, None)
    shared["cooldowns"][sym] = ctx.params.cooldown_bars
    ctx.close(sym, reason=f"[rsi-div] {reason}")


def _try_store_signal(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
) -> None:
    """Detect a fresh bullish divergence and stash it as a pending signal.

    Signal-level gates only — bullish price-RSI divergence, RSI below the
    ``rsi_floor``, and the EMA trend up. The volume/EMA/RSI confirmation is
    deliberately *not* evaluated here; it belongs on a later confirmation bar.
    """
    lows = o.low.to_array()
    rsi_arr = ctx.ta.rsi(sym, params.rsi_period).to_array()
    if len(lows) < 2 * params.pivot_lookback + 1:
        return
    div = detect_bullish_divergence(
        lows, rsi_arr, params.pivot_lookback, params.rsi_floor
    )
    if div is None:
        return
    cur_i, prev_i = div

    fast = _last(ctx.ta.ema(sym, params.fast_ma))
    slow = _last(ctx.ta.ema(sym, params.slow_ma))
    # The slow trend gate is optional (see ``use_trend_gate``). When off, the
    # pure oversold divergence is the whole signal; when on, EMA20 must sit
    # above EMA63 as well.
    if params.use_trend_gate and not trend_bullish(fast, slow):
        return

    shared["pending"][sym] = {
        "low_idx": cur_i,
        "swing_low": float(lows[cur_i]),
        "prev_low_idx": prev_i,
        "age": 0,
    }


def _age_pending(shared: dict, params: Params, sym: str) -> None:
    """Bump a pending signal's age; drop it once past ``max_confirm_latency``."""
    pending = shared["pending"][sym]
    pending["age"] += 1
    if pending["age"] > params.max_confirm_latency:
        del shared["pending"][sym]


def _try_confirm(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
) -> bool:
    """Evaluate the confirmation bar for a pending signal; enter on success.

    Confirmation requires ALL of: (a) close back above the fast EMA, (b) RSI
    below ``max_rsi_at_trigger`` (not chasing a vertical rip), (c) rising
    volume participation via :func:`trigger_confirmation`, and (d) the slow EMA
    trend still up **when ``use_trend_gate`` is enabled**. The ``close > EMA20``
    reclaim (a) and the RSI cap (b) are the volume/reversal-quality gates and
    are enforced regardless. On success emits a risk-sized long and records the
    position; returns True. Otherwise returns False (the pending is
    aged/dropped upstream).
    """
    pending = shared["pending"][sym]
    fast = _last(ctx.ta.ema(sym, params.fast_ma))
    slow = _last(ctx.ta.ema(sym, params.slow_ma))
    if params.use_trend_gate and not trend_bullish(fast, slow):  # (d)
        return False

    close = float(o.close[-1])
    if not (np.isfinite(fast) and close > fast):  # (a) reversed above EMA20
        return False

    rsi_arr = ctx.ta.rsi(sym, params.rsi_period).to_array()
    rsi_now = float(rsi_arr[-1])
    if not (np.isfinite(rsi_now) and rsi_now < params.max_rsi_at_trigger):  # (b)
        return False

    vol = o.volume.to_array()
    if not trigger_confirmation(vol, params.confirm_vol_window):  # (c)
        return False

    return _emit_entry(ctx, shared, params, sym, o, float(pending["swing_low"]))


def _emit_entry(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
    swing_low: float,
) -> bool:
    """ATR-risk-size a long from the confirmation close and record it.

    Mirrors the module's established sizing: stop distance from the swing low
    (floored at one ATR), ``risk_sized_qty`` on current equity, back-solved
    ``size``, fractional SL/TP. Returns False if any guard fails (no emit).
    """
    close = float(o.close[-1])
    atr = _last(ctx.ta.atr(sym, _ATR_PERIOD))
    if not np.isfinite(atr) or atr <= 0:
        return False

    stop_dist = stop_distance(close, swing_low, atr, params.stop_atr_mult)
    if stop_dist <= 0:
        return False

    qty = risk_sized_qty(
        equity=equity_of(ctx.state.portfolio),
        price=close,
        stop_dist=stop_dist,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return False
    size = entry_size(qty, close, ctx.state.portfolio.initial_capital)
    if size <= 0:
        return False

    sl_fraction = stop_dist / close
    tp_fraction = (params.take_profit_atr * atr) / close
    ctx.long(
        sym,
        size=size,
        sl=sl_fraction,
        tp=tp_fraction,
        tag="rsi-div",
        reason="[rsi-div] divergence confirmed",
    )
    shared["positions"][sym] = {"entry": close, "swing_low": swing_low, "pid": ""}
    return True
