"""Strategy-facing trend gates.

The regime layer hands strategies typed, cursor-safe trend gates computed
directly from ``state.candles``. The DSL replaces the old engine ``ModelState``
channel entirely: a strategy holds its own model object (e.g.
``src.indicators.hmm.strategy.OnlineRegime``) in ``ctx.shared`` and reads its
signal inline, instead of relying on an engine ``model_updater`` writing canned
fields to ``ModelState``.

Design:

* **Pure, cursor-safe** — every gate reads ``state.candles`` only (already
  truncated to the engine cursor, so no look-ahead) and returns a value; no
  mutable module state, no side effects.
* **Caller-owned cache** — the only expensive gate is ``weekly_above_sma``
  (a weekly resample). It accepts an optional ``cache`` dict so the *strategy*
  owns lifetime and reset semantics via its ``GLOBAL`` dict — no hidden global
  registry, no new engine wiring, and split/optimize reset stays correct.
* **Typed result** — ``TrendGate`` bundles a label plus boolean helpers so
  strategies stop threading "is it BULL/BEAR/RANGE/None" booleans by hand.

Sources with distinct trend concepts:

  =====================  ======================================================
  helper                 gates on
  =====================  ======================================================
  sma_trend              SMA fast/slow crossover (+ range threshold)
  above_sma              single price vs SMA (per symbol / proxy symbol)
  weekly_above_sma       weekly close vs weekly SMA (structural trend)
  =====================  ======================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.bt.regime.types import TrendRegime
from src.bt.state import BacktestState

# Caching key separator for strategy-owned cache dicts.
_CACHE_SEP = "\x00"


def _closes(state: BacktestState, symbol: str, bar: str) -> pd.Series | None:
    """Fetch close series for a symbol's interval (cursor-safe).

    ``bar`` is a candle interval (e.g. "1d"), or a higher timeframe than the
    traded bar (e.g. "1d" while trading "1h"). Safe on missing symbols.
    """
    df = state.candles.get((symbol, bar))
    if df is None or not len(df) or "close" not in df.columns:
        return None
    return df["close"]


@dataclass(frozen=True)
class TrendGate:
    """Current trend regime for one symbol/interval.

    ``label`` is None until enough bars exist to classify (warmup). Boolean
    helpers are the explicit gate a strategy applies to its candle.
    """

    label: TrendRegime | None

    @property
    def known(self) -> bool:
        return self.label is not None

    @property
    def bull(self) -> bool:
        return self.label == "BULL"

    @property
    def bear(self) -> bool:
        return self.label == "BEAR"

    @property
    def range(self) -> bool:
        return self.label == "RANGE"

    def allows_long(
        self,
        allow_bull: bool = True,
        allow_range: bool = False,
        allow_unknown: bool = True,
    ) -> bool:
        """True if this gate permits a long entry.

        Defaults: long only in a known BULL. BEAR never allows a long. RANGE
        allows one only when ``allow_range`` is True (e.g. a strategy that
        fades ranges instead of standing aside). ``allow_unknown`` governs the
        pre-warmup None label.
        """
        if not self.known:
            return allow_unknown
        if self.bull:
            return allow_bull
        if self.range:
            return allow_range
        return False

    def allows_short(
        self,
        allow_bear: bool = True,
        allow_range: bool = False,
        allow_unknown: bool = True,
    ) -> bool:
        """True if this gate permits a short entry.

        Defaults: short only in a known BEAR. BULL never allows a short. RANGE
        allows one only when ``allow_range`` is True.
        """
        if not self.known:
            return allow_unknown
        if self.bear:
            return allow_bear
        if self.range:
            return allow_range
        return False

    def hostile_to(self, direction: str, allow_range: bool = False) -> bool:
        """True when the regime has turned against an open position.

        Used for regime-based exits. A long is hostile BEAR; a short is hostile
        BULL. RANGE is hostile only when ``allow_range`` is True.
        """
        if not self.known:
            return False
        if self.range:
            return allow_range
        if direction == "long":
            return self.bear
        if direction == "short":
            return self.bull
        return False

    def __str__(self) -> str:
        return self.label or "?"


def sma_trend(
    state: BacktestState,
    symbol: str,
    fast: int = 50,
    slow: int = 200,
    bar: str = "1d",
    range_threshold_pct: float = 0.005,
) -> TrendGate:
    """Classify BULL/BEAR/RANGE via fast/slow SMA crossover.

    The single shared formula behind every SMA-cross trend gate:
      - BULL: fast SMA > slow SMA and spread beyond the range threshold
      - BEAR: fast SMA < slow SMA and spread beyond the range threshold
      - RANGE: fast/slow spread within the threshold band (sideways)

    ``bar`` selects the timeframe to classify (e.g. "1d" trend while trading
    "1h"; default "1d").
    """
    closes = _closes(state, symbol, bar)
    if closes is None or len(closes) < slow:
        return TrendGate(None)

    fast_sma = float(closes.rolling(fast).mean().iloc[-1])
    slow_sma = float(closes.rolling(slow).mean().iloc[-1])
    if slow_sma <= 0 or pd.isna(fast_sma) or pd.isna(slow_sma):
        return TrendGate(None)

    spread = abs(fast_sma - slow_sma) / slow_sma
    if spread <= range_threshold_pct:
        label: TrendRegime = "RANGE"
    elif fast_sma > slow_sma:
        label = "BULL"
    else:
        label = "BEAR"
    return TrendGate(label)


def series_above_sma(closes: pd.Series, window: int) -> bool:
    """True if the last close is above its ``window``-SMA.

    Stateless variant for callers that already hold a close series (e.g. a
    proxy loaded externally rather than from ``CandleStore``). Returns False on
    insufficient data so warmup is a clean no-trade.
    """
    if closes is None or len(closes) < window + 1:
        return False
    sma = float(closes.rolling(window).mean().iloc[-1])
    last = float(closes.iloc[-1])
    if pd.isna(sma):
        return False
    return last > sma


def above_sma(
    state: BacktestState,
    symbol: str,
    window: int,
    bar: str = "1d",
) -> bool:
    """True if the symbol's last close is above its ``window``-SMA.

    The generic structural-trend test — the basis of the sector bear gate
    (proxy symbol close < SMA) and the daily trend filter. No classification,
    just the boolean, so callers decide direction.
    """
    closes = _closes(state, symbol, bar)
    if closes is None:
        return False
    return series_above_sma(closes, window)


def weekly_above_sma(
    state: BacktestState,
    symbol: str,
    window: int = 50,
    bar: str = "1d",
    min_weekly_return: float = -99.0,
    cache: dict | None = None,
) -> bool:
    """True if the symbol's weekly structure is bullish (close > weekly SMA).

    Structural-trend filter used by the pullback strategies: buckets the daily
    closes into weeks of 5 trading days (``resample`` is expensive and avoided)
    and tests the latest weekly close > its ``window``-SMA. Long-only setups
    skip entry while the weekly structure is below its SMA.

    ``min_weekly_return`` is an optional additional gate requiring the latest
    weekly return to exceed a fraction (e.g. 0.01); ``-99.0`` (default) disables
    it, matching the pullback strategies' effective config.

    ``cache`` is caller-owned (hand the strategy's ``GLOBAL`` dict) and keyed
    by the last daily bar's ISO week, so the weekly series is only recomputed
    when a new week actually starts. Like the previous resampled gate the
    caller controls lifetime and reset, exactly like its other caches.
    """
    closes = _closes(state, symbol, bar)
    if closes is None or len(closes) < window * 5 + 1:
        return False

    key = f"{symbol}{_CACHE_SEP}{bar or ''}{_CACHE_SEP}{window}"
    iso = closes.index[-1].isocalendar()
    wk = (iso.year, iso.week)

    if cache is not None and cache.get(key) is not None:
        entry: dict = cache[key]
        if entry["week"] == wk:
            # Same trading week: only the current bucket moved.
            weekly = list(entry["vals"])
            weekly[0] = float(closes.iloc[-1])
        else:
            # New week: bucket forward and shift, keep the window.
            weekly = _weekly_vals(closes, window)
            cache[key] = {"week": wk, "vals": weekly}
    else:
        weekly = _weekly_vals(closes, window)
        if cache is not None:
            cache[key] = {"week": wk, "vals": weekly}

    sma = float(sum(weekly[:window]) / window)
    result = float(weekly[0]) > sma

    if min_weekly_return > -99.0 and len(weekly) >= 2:
        weekly_ret = (weekly[0] - weekly[1]) / weekly[1]
        result = result and float(weekly_ret) > min_weekly_return

    return result


def _weekly_vals(closes: pd.Series, window: int) -> list[float]:
    """Last ``window`` weekly buckets (5 trading days each), newest first.

    No ``resample``: buckets are 5 contiguous trading days taken from the tail,
    so the current week is always the most recent 5 closes and each value is
    that bucket's final close (matching ``resample('W').last()`` semantics).
    """
    n = window * 5
    arr = closes.iloc[-n:].to_numpy()
    arr = arr.reshape(window, 5)
    # Every bucket's last close, newest week first.
    return [float(arr[-i - 1][-1]) for i in range(window)]
