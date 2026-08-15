"""Strategy-facing online Volume Profile owner.

``OnlineVP`` is the strategy-level owner of the accumulating Volume Profile —
the boolean-hot-path companion to ``OnlineVolumeProfile`` that reads real
OHLCV from ``CandleStore`` so a strategy can hold one per symbol in
``ctx.shared`` and call ``observe()`` once per candle.

Usage (inside a strategy's ``on_candle``, DSL stateful)::

    # Construct once per symbol, keyed in ctx.shared (fresh per run/window):
    shared = ctx.shared
    prof = shared.setdefault(
        "vp", {sym: OnlineVP(window=200) for sym in ctx.symbols}
    )

    for sym in ctx.symbols:
        snap = prof[sym].observe(ctx.state, sym, ctx.interval)
        if not snap.ready:
            continue  # warmup
        close = ctx.ohlcv(sym).close[-1]
        if close > snap.vah:
            ...  # breakout above the value area
        elif close < snap.val:
            ...  # breakdown below the value area
        elif snap.val <= close <= snap.vah:
            ...  # inside the value area (range)
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.indicators.volume_profile.online import OnlineVolumeProfile
from src.indicators.volume_profile.types import VolumeProfileSnapshot


class OnlineVP:
    """Owns one ``OnlineVolumeProfile`` that feeds off ``CandleStore``.

    Pass the engine ``BacktestState`` together with a symbol and interval and
    :meth:`observe` reads that symbol's latest cursor-safe candle and folds it
    into the running profile. Use one instance per symbol (or key by symbol in
    ``ctx.shared``). Reconstruct per run/window to keep split/sweep isolated.

    Parameters mirror :class:`OnlineVolumeProfile`.
    """

    def __init__(
        self,
        num_bins: int = 50,
        value_area_pct: float = 0.70,
        window: int | None = 200,
        warmup_bars: int = 50,
    ) -> None:
        self._profile = OnlineVolumeProfile(
            num_bins=num_bins,
            value_area_pct=value_area_pct,
            window=window,
            warmup_bars=warmup_bars,
        )

    @property
    def profile(self) -> OnlineVolumeProfile:
        """The underlying accumulator (for diagnostics / reset)."""
        return self._profile

    @property
    def n_observed(self) -> int:
        return self._profile.n_observed

    def observe(
        self,
        state,
        symbol: str,
        interval: str,
    ) -> VolumeProfileSnapshot:
        """Feed ``symbol``'s latest candle at ``interval`` into the profile.

        Reads the newest cursor-safe bar for ``(symbol, interval)`` from
        ``state.candles`` and calls :meth:`OnlineVolumeProfile.observe`. If the
        symbol has no candles yet, returns the current snapshot unchanged.
        """
        df = state.candles.get((symbol, interval))
        if df is None or len(df) < 1:
            return self._profile.snapshot()

        last = cast(pd.Series, df.iloc[-1])
        high = float(last["high"])
        low = float(last["low"])
        volume = float(last["volume"])
        return self._profile.observe(low, high, volume)
