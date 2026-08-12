"""Pine-flavoured declarative strategy framework (Option A of prompt2.md).

A decorated strategy is a tiny pure function of a :class:`StrategyContext` --
it describes *what* to do per candle, not *how* the engine delivers data. The
framework owns the plumbing the raw ``on_candle`` style re-invents everywhere:

* candle iteration, cursor advancement, per-symbol signal bucketing,
* OHLCV + indicator prefetch into a cursor-safe ``ctx.ta`` (``TaContext``),
* the ``GLOBAL``/``reset_global`` lifecycle via an implicit ``ctx.state``,
* signal construction from ``ctx.long/close/...``.

A decorated strategy still exposes ``on_candle(state, candle, params)`` +
``STRATEGY_TYPE`` + a generated ``reset_global``, so it plugs straight into the
existing auto-discovery, ``bt split`` and ``bt sweep`` -- no engine fork.

Shape::

    STRATEGY_TYPE = "ema_cross"

    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext):
        fast = ctx.ta.ema("AAPL", 9)
        slow = ctx.ta.ema("AAPL", 21)
        if ctx.cross_over(fast, slow):
            ctx.long("AAPL", size=0.1, sl=0.04, tp=0.08)
        elif ctx.cross_under(fast, slow):
            ctx.close("AAPL")

The decorator is a pure adapter: it wraps the plain decision function in an
``on_candle(state, candle, params)`` that builds a ``StrategyContext``,
collects the signals the user's ``ctx.long/close`` calls emit, and returns
them. The raw ``on_candle`` hook stays intact for non-DSL power users.
"""

from __future__ import annotations

import sys
from types import FunctionType
from typing import Any, Callable, TYPE_CHECKING

from src.bt.state import ActionType, TradeSignal
from src.bt.strategies.series import SeriesView
from src.bt.strategies.ta_context import OhlcvView, TaContext
from src.bt.strategies.utils import sl_tp_from_pct

if TYPE_CHECKING:
    from src.bt.state import BacktestState, Candle, Position


class StrategyContext:
    """Per-candle decision surface handed to a decorated strategy.

    Every data access is cursor-safe: ``ctx.ohlcv`` and ``ctx.ta`` read through
    the engine's ``TaContext``, which shares the ``CandleStore`` cursor and can
    never expose a future bar. ``ctx.state`` is the raw ``BacktestState`` for
    power users (portfolio / position lookup) -- the DSL does not forbid it, it
    just makes the common path safe by construction.

    Sizing: ``size`` is a 0..1 fraction of *initial* capital converted to an
    absolute share count (``size * initial_capital / close``) — a fixed-percent
    order, not scaled by available cash. ``sl``/``tp`` are fractional
    percentages (e.g. ``0.04`` = 4%) converted to absolute per-trade stop/target
    prices.
    """

    __slots__ = (
        "_state",
        "_candle",
        "_ta",
        "_params",
        "_symbols",
        "_interval",
        "_signals",
        "_shared",
    )

    def __init__(
        self,
        state: "BacktestState",
        candle: "Candle",
        params: Any,
        ta: TaContext,
        symbols: tuple[str, ...],
        interval: str,
    ) -> None:
        self._state = state
        self._candle = candle
        self._ta = ta
        self._params = params
        self._symbols = symbols
        self._interval = interval
        self._signals: list[TradeSignal] = []
        self._shared: dict | None = None

    @property
    def shared(self) -> dict:
        """Framework-owned cross-cancel storage (only when the strategy is
        declared ``@strategy(stateful=True)``). Persists across candles within
        a run and is cleared by ``reset_global()`` between split/sweep windows.

        Raises when the strategy wasn't declared stateful -- calling this from a
        stateless strategy is a footgun, so fail loudly rather than silently
        sharing nothing.
        """
        if self._shared is None:
            raise RuntimeError(
                "ctx.shared requires the strategy to be declared "
                "`@strategy(stateful=True)` (cross-call state is the DSL's "
                "GLOBAL replacement)."
            )
        return self._shared

    @shared.setter
    def shared(self, holder: dict) -> None:
        """Bind the framework-owned state holder (the adapter wires this for
        stateful strategies). Mirrors the ``shared`` getter so the adapter
        doesn't poke ``self._shared`` directly.
        """
        self._shared = holder

    # -- public read-only accessors ------------------------------------------

    @property
    def state(self) -> "BacktestState":
        return self._state

    @property
    def candle(self) -> "Candle":
        return self._candle

    @property
    def timestamp(self):
        return self._candle.timestamp

    @property
    def params(self) -> Any:
        return self._params

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def ta(self) -> TaContext:
        return self._ta

    # -- data access -----------------------------------------------------------

    def ohlcv(self, sym: str, interval: str | None = None) -> OhlcvView:
        return self._ta.ohlcv(sym, interval or self._interval)

    def price(self, sym: str) -> float:
        """Current close for ``sym`` (cursor-safe O(1))."""
        return self._ta.close(sym, self._interval)[-1]

    def position(self, sym: str) -> "Position | None":
        """In-position? Returns the newest open :class:`Position` for ``sym``."""
        tup = self._state.portfolio.positions.get(sym)
        if not tup:
            return None
        return tup[-1]

    # -- signal emission ---------------------------------------------------------

    def long(
        self,
        sym: str,
        size: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        reason: Any = "long",
    ) -> None:
        """Open a long position in ``sym`` ``size`` fraction of capital.

        ``size`` is a 0..1 fraction of *initial* capital converted to an
        absolute share count (``size * initial_capital / price``) before
        emission — a fixed-size order, not scaled by available cash. When
        omitted, defaults to the engine's ``config.position_size``. ``sl``/
        ``tp`` are fractional percentages converted to absolute levels.
        """
        self._emit(ActionType.long, sym, size, sl, tp, reason)

    def short(
        self,
        sym: str,
        size: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        reason: Any = "short",
    ) -> None:
        """Open a short position in ``sym`` ``size`` fraction of capital.

        ``size`` is a 0..1 fraction of *initial* capital converted to an
        absolute share count (``size * initial_capital / price``) before
        emission — a fixed-size order, not scaled by available cash. When
        omitted, defaults to the engine's ``config.position_size``. ``sl``/
        ``tp`` are fractional percentages converted to absolute levels.
        """
        self._emit(ActionType.short, sym, size, sl, tp, reason)

    def close(self, sym: str, reason: Any = "close") -> None:
        """Close the open position in ``sym`` (full position).

        A no-op when ``sym`` is flat: closing a symbol with no position would
        otherwise emit a dangling ``close`` signal with ``position_id=None``
        that the engine must swallow. Guarding here keeps the emitted surface
        clean and the strategy's intent ("close whatever we hold") explicit.
        """
        if self.position(sym) is None:
            return
        price = self.price(sym)
        pos = self.position(sym)
        assert pos is not None
        self._signals.append(
            TradeSignal(
                action=ActionType.close,
                symbol=sym,
                timestamp=self._candle.timestamp,
                price=price,
                reason=reason,
                position_id=pos.position_id,
            )
        )

    def _emit(
        self,
        action: ActionType,
        sym: str,
        size: float | None,
        sl: float | None,
        tp: float | None,
        reason: Any,
    ) -> None:
        price = self.price(sym)
        is_long = action == ActionType.long
        sl_price, tp_price = sl_tp_from_pct(
            price, sl or 0.0, tp or 0.0, is_long=is_long
        )
        # ``size`` is a fixed 0..1 fraction of *initial* capital -> absolute
        # share count. Not scaled by available cash (a fixed-size order).
        qty = (
            0.0
            if size is None
            else round(size * self._state.portfolio.initial_capital / price, 4)
        )
        self._signals.append(
            TradeSignal(
                action=action,
                symbol=sym,
                timestamp=self._candle.timestamp,
                price=price,
                qty=qty,
                reason=reason,
                stop_loss=sl_price,
                take_profit=tp_price,
            )
        )

    # -- Pine built-ins (pure; operate on cursor-truncated views / floats) ------

    def nz(self, v: float, fallback: float = 0.0) -> float:
        return v if v == v else fallback

    def cross_over(self, a, b) -> bool:
        return _cross(a, b, over=True)

    def cross_under(self, a, b) -> bool:
        return _cross(a, b, over=False)

    def change(self, series: SeriesView, bars: int = 1) -> float:
        return series.change(bars)

    def barssince(self, pred: Callable[[int], bool], max_bars: int = 500) -> float:
        """Pine ``barssince`` — bars ago the ``pred(offset)`` was last True.

        ``pred(i)`` tests the bar ``i`` bars in the past (``i=0`` = current
        candle). Returns NaN when ``pred`` is never True within ``max_bars``.
        """
        for i in range(max_bars):
            if pred(i):
                return float(i)
        return float("nan")


# ---------------------------------------------------------------------------
# decorator + wrapper
# ---------------------------------------------------------------------------


class _StrategyAdapter:
    """Callable engine hook produced by ``@strategy``.

    Wraps a user ``def on_candle(ctx)`` so it exposes the engine contract
    ``on_candle(state, candle, params) -> list[TradeSignal]`` while building a
    cursor-safe :class:`StrategyContext` for the decision body. Created per
    decoration; holds the cross-call state holder (when stateful).
    """

    __slots__ = (
        "ctx_fn",
        "params_cls",
        "interval",
        "stateful",
        "__name__",
    )

    def __init__(
        self,
        fn: FunctionType,
        bars: str,
        stateful: bool,
    ) -> None:
        self.ctx_fn = fn
        self.params_cls = None
        self.interval = bars
        self.stateful = stateful
        self.__name__ = "on_candle"

    def reset(self) -> None:
        """Wired as the module's ``reset_global`` for split/sweep back-compat.

        With per-run state holders (minted fresh by the engine for every run
        window), there is no module-level dict to clear — a new window simply
        gets a new holder. Kept as a no-op so old ``bt split``/``bt sweep``
        callers that invoke ``reset_global()`` between windows keep working
        without racing on shared state.
        """
        return None

    def __call__(
        self,
        state: "BacktestState",
        candle: "Candle",
        params: Any,
    ) -> list[TradeSignal]:
        ta = getattr(state.candles, "ta", None)
        if not isinstance(ta, TaContext):
            raise RuntimeError(
                "DSL strategy requires a prefetched TaContext; run through "
                "`src.bt.engine.backtest.run` (it builds `ta` from data) so "
                f"state.candles.ta is set for module {self.ctx_fn.__module__}."
            )
        holder = None
        if self.stateful:
            holder = getattr(state.candles, "strategy_state", None)
            if not isinstance(holder, dict):
                raise RuntimeError(
                    "Stateful DSL strategy requires a per-run state holder; run "
                    "through `src.bt.engine.backtest.run` (it mints a fresh "
                    "holder per window) so `state.candles.strategy_state` is a "
                    f"dict for module {self.ctx_fn.__module__}."
                )
        ctx = StrategyContext(
            state=state,
            candle=candle,
            params=params,
            ta=ta,
            symbols=symbols_from(state),
            interval=self.interval,
        )
        if holder is not None:
            ctx.shared = holder
        self.ctx_fn(ctx)
        return ctx._signals


def strategy(bars: str = "1d", stateful: bool = False):
    """Decorate ``def on_candle(ctx: StrategyContext)`` into an engine hook.

    Returns an adapter (callable ``on_candle(state, candle, params) ->
    list[TradeSignal]``) wrapping the plain decision function with a
    ``StrategyContext``. The raw ``on_candle`` hook stays intact for non-DSL
    power users.

    Cross-candle state (``stateful=True``) is **per-run**: the engine mints a
    fresh holder for every ``run``/window and the adapter reads it from
    ``state.candles.strategy_state``. Nothing is shared at module scope, so a
    stateless OR stateful DSL strategy is thread-safe across concurrent
    ``run_split``/``run_sweep``/``run_optimize`` workers. ``reset_global`` is
    kept as a no-op shim for split/sweep back-compat.

    Args:
        bars: signal interval served by ``ctx`` (matches the config base bar).
        stateful: when True, persist cross-call state in ``ctx.shared`` (a
            per-run dict, fresh for every run window).
    """

    def decorate(fn: FunctionType):
        adapter = _StrategyAdapter(fn, bars, stateful)
        module = sys.modules.get(fn.__module__)
        if module is not None:
            adapter.params_cls = getattr(module, "Params", None)
            strategy_type = getattr(module, "STRATEGY_TYPE", None)
            if strategy_type is None:
                strategy_type = getattr(fn, "STRATEGY_TYPE", None)
            if strategy_type is not None:
                setattr(module, "STRATEGY_TYPE", strategy_type)
            # reset_global is a no-op back-compat shim. State is per-run (minted
            # fresh by the engine per window), so cross-window bleed is already
            # impossible without any module-level clear.
            setattr(module, "reset_global", adapter.reset)
        return adapter

    return decorate


def symbols_from(state: "BacktestState") -> tuple[str, ...]:
    """Symbols present in the candle store's accumulator (deduped, ordered).

    Dedups by symbol across every interval key (base + any HTF), preserving the
    store's deterministic insertion order. O(S) via a set, not an O(S²) list scan.
    By the time ``on_candle`` fires, the store is populated for all configured
    symbols, so this is authoritative at call sites.
    """
    seen: set[str] = set()
    result: list[str] = []
    for k in state.candles.keys():
        sym = k[0]
        if sym not in seen:
            seen.add(sym)
            result.append(sym)
    return tuple(result)


# ---------------------------------------------------------------------------
# cross helpers
# ---------------------------------------------------------------------------


def _read_position(v) -> float:
    return v[-1] if isinstance(v, SeriesView) else float(v)


def _read_previous(v) -> float:
    return v[-2] if isinstance(v, SeriesView) else float(v)


def _cross(a, b, over: bool) -> bool:
    """True when ``a`` crossed ``b`` on the current bar in the given direction.

    SeriesViews read the current + previous cursor-truncated values (O(1), no
    lookahead); raw floats compare against themselves (a degenerate, usually
    false, single-value cross).
    """
    a_cur, b_cur = _read_position(a), _read_position(b)
    a_prev, b_prev = _read_previous(a), _read_previous(b)
    if over:
        return a_prev <= b_prev and a_cur > b_cur
    return a_prev >= b_prev and a_cur < b_cur


__all__ = [
    "strategy",
    "StrategyContext",
    "SeriesView",
    "OhlcvView",
    "TaContext",
    "symbols_from",
]
