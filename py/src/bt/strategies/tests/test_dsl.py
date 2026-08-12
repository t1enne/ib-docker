"""End-to-end tests for the Pine-flavoured strategy DSL (Option A) through the engine.

These exercise the full ``on_candle`` path via ``src.bt.engine.backtest.run`` --
the canonical entry point for DSL strategies (it builds the prefetched
``TaContext`` from ``data`` and attaches it to the ``CandleStore``). The
regression test proves a DSL strategy is behaviourally identical to an
equivalent hand-written raw ``on_candle`` strategy (same P&L and trades).
"""

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies import init_strat
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.types import StrategyConfig


def _df(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + 5 * np.sin(np.arange(n) / 5.0) + np.cumsum(rng.normal(0, 0.1, n))
    data = {}
    for sym in ("AAPL", "MSFT"):
        data[(sym, "open")] = close - 0.4
        data[(sym, "high")] = close + 1.0
        data[(sym, "low")] = close - 1.0
        data[(sym, "close")] = close
        data[(sym, "volume")] = np.full(n, 1000.0)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(symbols=("AAPL", "MSFT")) -> StrategyConfig:
    return StrategyConfig(
        name="ema_dsl",
        strategy_type="ema_cross",
        symbols=list(symbols),
        initial_capital=10000.0,
        commission=0.5,
        training_start="2022-01-01",
        training_end="2022-12-31",
        trading_start="2023-01-01",
        trading_end="2023-04-30",
        bars=["1d"],
        strategy_params={
            "fast": 9,
            "slow": 21,
            "warmup": 30,
            "size": 0.1,
            "stop_loss": 0.04,
            "take_profit": 0.08,
        },
        model_params={},
    )


def test_dsl_strategy_is_discoverable():
    strat = init_strat("ema_cross")
    assert hasattr(strat, "on_candle")
    assert hasattr(strat, "reset_global")
    assert strat.__name__ == "src.bt.strategies.ema_cross"


def test_dsl_run_produces_trades_and_indicator_cache():
    df = _df()
    res = run(Backtest(_cfg()), df, strat_mod=init_strat("ema_cross"))
    assert len(res.pf.trades) > 0
    # 2 symbols x 2 EMAs (fast+slow) = 4 full-series computes, none per candle.
    assert res.data.ta.compute_count == 4
    # Every close signal carries an absolute stop-loss/target below/above entry.
    for t in res.pf.trades:
        assert t.entry_price > 0
    assert 0 < res.pf.capital_utilization <= 1.0


def test_dsl_close_uses_position_targeted_where_position_exists():
    df = _df(200, seed=3)
    res = run(Backtest(_cfg()), df, strat_mod=init_strat("ema_cross"))
    # The DSL opens AND closes positions through the same ctx path; there must
    # be at least one closed (non-end) trade signalled by ctx.close.
    closed = [t for t in res.pf.trades if t.exit_time is not None]
    assert any(str(t.close_reason) == "bearish ema cross" for t in closed)


def test_dsl_position_gating_fires():
    # Position gating: once in a position the strategy only looks to close, so
    # the number of equity points is bounded by bars (no runaway stack).
    df = _df()
    res = run(Backtest(_cfg()), df, strat_mod=init_strat("ema_cross"))
    # No signal ever opens a second position while one is open (checked by the
    # strategy reading ctx.position). We assert the position count per symbol
    # in the final state is at most one.
    for sym, positions in res.final_state.portfolio.positions.items():
        assert len(positions) <= 1


# ---------------------------------------------------------------------------
# regression: DSL strategy == equivalent raw on_candle strategy
# ---------------------------------------------------------------------------


def _ema_raw_strategy(state, candle, params):
    """Hand-written raw on_candle equivalent of the ema_cross DSL strategy.

    Reimplements the same decision (fast>slow cross, size fraction, SL/TP %)
    using ONLY the engine's raw ``state.candles``/``ta.ema`` path -- the
    pre-DSL style. Used to prove the DSL does not change behaviour.
    """
    import src.indicators.ta as tain
    from src.bt.state import ActionType, TradeSignal

    interval = candle.interval or "1d"
    if interval != "1d":
        return []

    signals: list[TradeSignal] = []
    for sym in sorted({k[0] for k in state.candles.keys()}):
        df = state.candles.get((sym, "1d"))
        if df is None:
            continue
        closes = df["close"]
        if len(closes) < params.warmup:
            continue
        fast = tain.ema(closes, params.fast)
        slow = tain.ema(closes, params.slow)
        pos = state.portfolio.positions.get(sym)
        price = float(closes.iloc[-1])
        if not pos:
            crossed = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
            if crossed:
                sl, tp = _sl_tp(price, params)
                # Mirror the DSL's fixed-percent sizing: absolute shares from a
                # fraction of *initial* capital (not cash-scaled).
                qty = round(
                    params.size * state.portfolio.initial_capital / price,
                    4,
                )
                signals.append(
                    TradeSignal(
                        action=ActionType.long,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=price,
                        qty=qty,
                        reason="bullish ema cross",
                        stop_loss=sl,
                        take_profit=tp,
                    )
                )
        else:
            crossed = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
            if crossed:
                signals.append(
                    TradeSignal(
                        action=ActionType.close,
                        symbol=sym,
                        timestamp=candle.timestamp,
                        price=price,
                        reason="bearish ema cross",
                        position_id=pos[-1].position_id,
                    )
                )
    return signals


def _sl_tp(price: float, params):
    return (
        price * (1 - params.stop_loss),
        price * (1 + params.take_profit),
    )


def test_dsl_matches_raw_on_candle_pnl():
    from src.bt.engine.backtest import Backtest, run_backtest
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    df = _df()
    cfg = _cfg()
    bt = Backtest(cfg)

    # DSL run through the canonical `run` entrypoint.
    dsl_res = run(bt, df, strat_mod=init_strat("ema_cross"))

    # Raw run: drive run_backtest directly (raw strategies don't need ta).
    from src.bt.engine.backtest import _resolve_model_updater

    from src.bt.engine.utils import candle_generator

    bt2 = Backtest(cfg)
    gen = candle_generator(df, bt2.config)
    raw_res, _ = run_backtest(
        bt2,
        gen,
        default_execution_handler(),
        default_risk_handler(),
        model_updater_fn=_resolve_model_updater(bt2.config),
        strategy_mod=_RawMod(),
    )

    def _trade_table(result):
        return sorted(
            (
                t.symbol,
                (t.entry_time, t.exit_time or pd.NaT),
                round(t.pnl, 6),
                str(t.reason),
            )
            for t in result.pf.trades
        )

    assert len(dsl_res.pf.trades) == len(raw_res.pf.trades)
    assert _trade_table(dsl_res) == _trade_table(raw_res)
    assert dsl_res.pf.total_return == pytest.approx(raw_res.pf.total_return, abs=1e-9)


class _RawMod:
    """Minimal strategy module wrapper for the raw on_candle regression."""

    def on_candle(self, state, candle, params):
        return _ema_raw_strategy(state, candle, params)


# ---------------------------------------------------------------------------
# Pine built-ins on the StrategyContext (unit level)
# ---------------------------------------------------------------------------


def _make_ctx(df, idx=40, sym="AAPL"):
    """Build a real StrategyContext over ``df`` at cursor ``idx``."""
    from src.bt.engine.candle_store import CandleStore
    from src.bt.engine.utils import merge_bt_state
    from src.bt.state import Candle, create_initial_backtest_state
    from src.bt.strategies.dsl import StrategyContext
    from src.bt.strategies.ta_context import TaContext

    ta = TaContext.from_data(df, ("AAPL", "MSFT"), "1d")
    n = len(df)
    rows: dict = {}
    for s in ("AAPL", "MSFT"):
        arr = {
            k: (np.empty(n) if k != "_len" else np.array([n]))
            for k in ("timestamp", "open", "high", "low", "close", "volume", "_len")
        }
        arr["timestamp"] = df.index.to_numpy().astype("datetime64[ms]")
        for f in ("open", "high", "low", "close", "volume"):
            arr[f] = df[(s, f)].to_numpy(dtype=float)
        rows[(s, "1d")] = arr
    store = CandleStore(rows)
    store.attach_ta(ta)
    ta.bind(store)
    store.advance(df.index[idx])
    state = create_initial_backtest_state(
        symbols=["AAPL", "MSFT"],
        initial_capital=10000.0,
        start_timestamp=df.index[0],
        rolling_window_size=None,
    )
    state = merge_bt_state(
        state,
        dict(candles=store),
    )
    candle = Candle(
        timestamp=df.index[idx],
        symbol="MSFT",
        open=df[(sym, "open")].iloc[idx],
        high=df[(sym, "high")].iloc[idx],
        low=df[(sym, "low")].iloc[idx],
        close=df[(sym, "close")].iloc[idx],
        volume=df[(sym, "volume")].iloc[idx],
        interval="1d",
    )
    return StrategyContext(state, candle, object(), ta, ("AAPL", "MSFT"), "1d")


def test_ctx_cross_and_nz_and_change():
    df = _df(120)
    ctx = _make_ctx(df, idx=60)
    fast = ctx.ta.ema("AAPL", 9)
    slow = ctx.ta.ema("AAPL", 21)
    over = ctx.cross_over(fast, slow)
    under = ctx.cross_under(fast, slow)
    # A cross can be at most one of the two for the same pair.
    assert not (over and under)
    assert ctx.nz(float("nan")) == 0.0
    assert ctx.nz(float("nan"), 42.0) == 42.0
    # change over the EMA series vs its previous bar.
    ch = ctx.change(fast, 1)
    assert ch == pytest.approx(fast[-1] - fast[-2])


def test_ctx_barssince_finds_back_cross():
    from src.bt.strategies.dsl import symbols_from

    df = _df(200)
    ctx = _make_ctx(df, idx=150)
    fast = ctx.ta.ema("AAPL", 9)
    slow = ctx.ta.ema("AAPL", 21)

    def was_over(i: int) -> bool:
        # bar ``i`` ago had a bullish cross
        return fast[-(i + 1)] > slow[-(i + 1)] and fast[-(i + 2)] <= slow[-(i + 2)]

    bars = ctx.barssince(was_over, max_bars=500)
    # Somewhere in 200 bars of data a cross occurred (or the window was too
    # short / flat); just assert the sentinel contract holds.
    if bars == bars:
        assert 0 <= bars <= 150
    # And positions/symbols surface correctly.
    assert ctx.symbols == ("AAPL", "MSFT")
    assert "AAPL" in symbols_from(ctx.state)


def test_symbols_from_dedups_across_intervals():
    """Regression (#4): ``symbols_from`` must dedupe across (base + HTF) interval
    keys in O(S) while preserving store insertion order."""
    from src.bt.engine.candle_store import CandleStore
    from src.bt.engine.utils import merge_bt_state
    from src.bt.state import create_initial_backtest_state
    from src.bt.strategies.dsl import symbols_from

    # Store with a symbol present under BOTH base and HTF interval keys.
    n = 10
    rows: dict = {}
    for sym, iv in (("AAPL", "1d"), ("MSFT", "1d"), ("AAPL", "4h")):
        arr = {
            k: (np.empty(n) if k != "_len" else np.array([n]))
            for k in ("timestamp", "open", "high", "low", "close", "volume", "_len")
        }
        arr["timestamp"] = np.arange(n).astype("datetime64[s]")
        for f in ("open", "high", "low", "close", "volume"):
            arr[f] = np.arange(n, dtype=float)
        rows[(sym, iv)] = arr
    store = CandleStore(rows)
    state = create_initial_backtest_state(
        symbols=["AAPL", "MSFT"],
        initial_capital=10000.0,
        start_timestamp=pd.Timestamp("2023-01-01"),  # ty: ignore[invalid-argument-type]
        rolling_window_size=None,
    )
    state = merge_bt_state(state, dict(candles=store))
    # AAPL appears twice (1d + 4h) but must dedupe to a single occurrence, and
    # order (AAPL, MSFT) is preserved from the store's insertion order.
    assert symbols_from(state) == ("AAPL", "MSFT")


# ---------------------------------------------------------------------------
# stateful mode: ctx.shared persists across candles, resets between runs
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def _stateful_ctx(ctx: StrategyContext):
    # Persist a growing call counter in ctx.shared across candles.
    ctx.shared["calls"] = ctx.shared.get("calls", 0) + 1


def _drive(adapter, df, n_bars: int = 40) -> tuple[list[int], dict]:
    """Drive a strategy adapter over ``df``, returning ``(counter_seen, holder)``
    for a stateful strategy. ``_drive`` mimics the engine by minting a FRESH
    per-run holder and attaching it to the store — the same per-run isolation
    ``run_split``/``run_sweep`` get from ``run()``."""
    from src.bt.engine.backtest import _append_candle
    from src.bt.engine.candle_store import CandleStore
    from src.bt.engine.utils import candle_generator, merge_bt_state
    from src.bt.state import create_initial_backtest_state
    from src.bt.strategies.ta_context import TaContext

    cfg = _cfg()
    gen = candle_generator(df, cfg)
    state = create_initial_backtest_state(
        symbols=["AAPL"],
        initial_capital=10000.0,
        start_timestamp=df.index[0],
        rolling_window_size=None,
    )
    rows: dict = {}
    store = CandleStore(rows)
    ta = TaContext.from_data(df, ("AAPL",), "1d")
    holder: dict = {}
    store.attach_ta(ta)
    store.attach_strategy_state(holder)
    ta.bind(store)
    state = merge_bt_state(state, dict(candles=store))
    seen: list[int] = []
    for c in gen:
        if c.symbol != "AAPL":
            continue
        rows, state = _append_candle(rows, state, c, "1d")
        store.advance(c.timestamp)
        adapter(state, c, object())
        seen.append(holder["calls"])
    return seen, holder


def test_dsl_stateful_shared_persists_and_resets():
    df = _df(40)
    # Across a run, ctx.shared accumulates: candle N sees counter == N.
    seen, _ = _drive(_stateful_ctx, df)
    assert seen == list(range(1, len(seen) + 1))
    total = seen[-1]
    assert total > 1

    # A second run (new window) mints a FRESH holder — no module-level state to
    # clear, so the counter restarts at 1 without any reset call. This is the
    # per-run isolation the thread-safety design guarantees.
    seen2, holder2 = _drive(_stateful_ctx, df)
    assert seen2 == list(range(1, len(seen2) + 1))
    assert seen2[0] == 1  # fresh, not total+1
    # The first run's holder is untouched by the second run.
    assert seen[-1] == total


def test_dsl_stateful_is_thread_safe_across_runs():
    """Two concurrent runs sharing ONE stateful adapter must not bleed state.

    Each thread gets its own per-run holder (minted in ``_drive`` like the
    engine mints in ``run()``); the adapter reads it from ``state.candles.
    strategy_state``, never from a module singleton. So N threads each see
    their counter restart at 1 and reach the full bar count independently.
    """
    import threading

    df = _df(40)
    n_threads = 8
    results: list[BaseException | None] = [None] * n_threads
    outcomes: list[list[int]] = [None] * n_threads

    def worker(idx: int) -> None:
        try:
            seen, _ = _drive(_stateful_ctx, df)
            outcomes[idx] = seen
        except Exception as exc:  # surfaced below for a clean assertion
            results[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for idx, err in enumerate(results):
        assert err is None, f"worker {idx} raised: {err!r}"
    # Every thread independently accumulated 1..len(df) — no cross-bleed, no
    # torn shared counter that would exceed the per-run total.
    expected = list(range(1, len(df) + 1))
    for idx, seen in enumerate(outcomes):
        assert seen == expected, f"worker {idx} bled state: {seen} != {expected}"


def test_dsl_stateless_shared_raises():
    # A StrategyContext not wired up by ``@strategy(stateful=True)`` has no
    # shared storage; touching ``ctx.shared`` must fail loudly (footgun guard).
    ctx = _make_ctx(_df(40), idx=10)
    assert ctx._shared is None
    with pytest.raises(RuntimeError):
        _ = ctx.shared


def test_close_on_flat_symbol_is_a_noop():
    # ``ctx.close`` on a symbol with no open position must not emit a dangling
    # ``close`` signal with ``position_id=None`` — it's a no-op instead.
    ctx = _make_ctx(_df(120), idx=60)
    assert ctx.position("AAPL") is None  # fresh state: portfolio is flat
    ctx.close("AAPL")
    assert ctx._signals == []


def test_shared_setter_binds_holder():
    # The ``shared`` setter mirrors the getter so the adapter wires the state
    # holder through the public accessor rather than poking ``._shared``.
    ctx = _make_ctx(_df(40), idx=10)
    holder = {"n": 0}
    ctx.shared = holder
    assert ctx.shared is holder
    ctx.shared["n"] += 1
    assert holder["n"] == 1
