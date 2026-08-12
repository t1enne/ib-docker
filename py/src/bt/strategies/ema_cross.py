"""EMA cross demo strategy, written on the Pine-flavoured strategy DSL.

Migrates the classic ``ema_cross`` signal from the raw ``on_candle(state,
candle, params)`` boilerplate to a short, declarative, hard-to-get-wrong
:class:`StrategyContext` function. It demonstrates the pieces Option A + C
deliver:

* indicators prefetched ONCE at engine start and read at the cursor in O(1)
  (``ctx.ta.ema``), so this pays one EMA compute per (symbol, period) total,
* cursor-truncated OHLCV reads (``ctx.ohlcv``) with no DataFrame rebuild,
* declarative signal construction (``ctx.long`` / ``ctx.close``) with
  fractional-percent SL/TP converted to absolute levels,
* Pine built-ins (``ctx.cross_over``) with lookahead-safe semantics.

It still exposes ``on_candle`` + ``STRATEGY_TYPE`` + a generated
``reset_global``, so it plugs straight into auto-discovery, ``bt split`` and
``bt sweep`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "ema_cross"


# Plain-Pine-style params. ``strategy_params: {"fast": 9, "slow": 21}`` in the
# config resolves straight into this typed dataclass.
@dataclass(frozen=True)
class Params(StrategyParams):
    fast: int = 9
    slow: int = 21
    warmup: int = 60
    size: float = 0.1  # fraction of free cash per position
    stop_loss: float = 0.04
    take_profit: float = 0.08


@strategy(bars="1d")
def on_candle(ctx: StrategyContext):
    """Open a long on a bullish EMA cross, close on the bearish cross.

    Stateful position management is expressed purely: ``ctx.position(sym)``
    tells us whether we already hold the symbol (green / red / flat).
    """
    for sym in ctx.symbols:
        if len(ctx.ohlcv(sym).close) < ctx.params.warmup:
            continue

        fast = ctx.ta.ema(sym, ctx.params.fast)
        slow = ctx.ta.ema(sym, ctx.params.slow)

        has_pos = ctx.position(sym) is not None

        if has_pos:
            if ctx.cross_under(fast, slow):
                ctx.close(sym, reason="bearish ema cross")
            continue

        # has open position
        if ctx.cross_over(fast, slow):
            ctx.long(
                sym,
                size=ctx.params.size,
                sl=ctx.params.stop_loss,
                tp=ctx.params.take_profit,
                reason="bullish ema cross",
            )
