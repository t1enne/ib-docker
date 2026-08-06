"""Strategy-facing online HMM owner.

Complements ``MarketRegimeHMMOnline`` (the raw model) and the engine's
``create_dual_online_updater`` / ``create_hmm_online_updater`` model updaters.
Those wire a *single hidden* HMM through ``config.model_updater`` and write a
canned label to ``ModelState.current_regime`` — so a strategy can neither own its
own per-symbol instances nor drive them inline.

``OnlineRegime`` is the strategy-level counterpart: construct one in the
strategy's ``reset_global()``, hold it in ``GLOBAL``, and call ``observe()``
once per candle. It lazily maintains a per-symbol ``MarketRegimeHMMOnline``
behind a scoped ``O(1)`` latest-close read over ``CandleStore`` (cursor-safe, no
look-ahead). Re-constructing in ``reset_global()`` gives clean IS/OOS reset.

Usage (inside a strategy's ``on_candle``)::

    GLOBAL = {"hmm": OnlineRegime()}

    def reset_global() -> None:
        global GLOBAL
        GLOBAL = {"hmm": OnlineRegime()}

    def on_candle(state, candle, params):
        vol = GLOBAL["hmm"].observe(state, candle.symbol, candle.interval or "1h")
        if vol is None:
            return []  # warmup
        # vol: 0=low, 1=med, 2=high (vol-ranked)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.indicators.hmm.online import MarketRegimeHMMOnline
from src.bt.state import BacktestState


@dataclass(frozen=True)
class OnlineRegimeResult:
    """One observation's HMM output, decoded to labels.

    ``label`` is None until the model has fitted (warmup). ``int`` is the raw
    vol-ranked regime (0=low, 1=med, 2=high) so callers can size or gate.
    """

    value: int | None
    fitted: bool


class OnlineRegime:
    """Per-symbol online HMM, owned by a strategy.

    Maintains one ``MarketRegimeHMMOnline`` per symbol, fed from the latest
    candle close of that symbol's interval. Regimes are vol-ranked:
    0=low vol, 1=med vol, 2=high vol.

    Parameters mirror ``MarketRegimeHMMOnline``.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        window_size: int = 500,
        vol_window: int = 20,
        momentum_window: int = 10,
        retrain_interval: int = 50,
        random_state: int = 42,
    ) -> None:
        self._models: dict[str, MarketRegimeHMMOnline] = {}
        self._n_regimes = n_regimes
        self._window_size = window_size
        self._vol_window = vol_window
        self._momentum_window = momentum_window
        self._retrain_interval = retrain_interval
        self._random_state = random_state

    def observe(
        self,
        state: BacktestState,
        symbol: str,
        interval: str,
    ) -> OnlineRegimeResult:
        """Feed the symbol's latest close to its HMM and return the regime.

        No-op (empty / warmup result) until enough closes exist. Reads only
        ``state.candles`` (Mapping interface, cursor-truncated) so it is
        look-ahead-safe.
        """
        df = state.candles.get((symbol, interval))
        if df is None or not len(df) or "close" not in df.columns:
            return OnlineRegimeResult(None, False)
        close = float(df["close"].iloc[-1])

        model = self._models.get(symbol)
        if model is None:
            model = MarketRegimeHMMOnline(
                n_regimes=self._n_regimes,
                window_size=self._window_size,
                vol_window=self._vol_window,
                momentum_window=self._momentum_window,
                retrain_interval=self._retrain_interval,
                random_state=self._random_state,
            )
            self._models[symbol] = model

        value = model.update(float(close))
        if value < 0:
            return OnlineRegimeResult(None, False)
        return OnlineRegimeResult(int(value), model.fitted)

    def n_steps(self, symbol: str) -> int:
        """Observations processed so far for a symbol (0 if never fed)."""
        model = self._models.get(symbol)
        return model.n_steps if model is not None else 0
