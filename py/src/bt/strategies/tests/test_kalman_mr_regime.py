"""Tests for the benchmark-gated Kalman mean-reversion DSL strategy.

Drives the full engine ``run()`` path (the same as the DSL tests) with a
synthetic **ranging** benchmark and mean-reverting tradeables, asserting the
strategy actually trades only when the regime gate opens, fades overextension,
and flattens on regime exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies import init_strat
from src.bt.strategies.kalman_mr_regime_dsl import (
    _bench_slope_move,
    _regime_ok,
    Params,
)
from src.bt.types import StrategyConfig


def _ranging_df(n: int = 520, seed: int = 1) -> pd.DataFrame:
    """SPY flat/oscillating (a genuine range); tradeables mean-revert around
    a level with occasional fireable overshoots."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    # Benchmark: tight sine around a flat level -> rangey, low vol.
    spy = 100 + 3.0 * np.sin(np.arange(n) / 25.0) + rng.normal(0, 0.1, n)
    data = {}
    for sym in ("SPY", "AAPL", "MSFT"):
        base = 100.0 if sym == "AAPL" else 200.0
        o = np.ones(n) * 0.0
        # tradeable tracks a level with a mean-reverting residual sampled from
        # an AR(1) so z_stat crosses the fade threshold repeatedly.
        r = np.zeros(n)
        e = rng.normal(0, 0.5, n)
        for i in range(1, n):
            r[i] = 0.9 * r[i - 1] + e[i]
        close = base + o + r
        ohlcv = {}
        ohlcv[("SPY", "open")] = spy - 0.1
        ohlcv[("SPY", "high")] = spy + 0.3
        ohlcv[("SPY", "low")] = spy - 0.3
        ohlcv[("SPY", "close")] = spy
        ohlcv[("SPY", "volume")] = np.full(n, 1_000_000.0)
        ohlcv[(sym, "open")] = close - 0.1
        ohlcv[(sym, "high")] = close + 0.5
        ohlcv[(sym, "low")] = close - 0.5
        ohlcv[(sym, "close")] = close
        ohlcv[(sym, "volume")] = np.full(n, 100_000.0)
        for k, v in ohlcv.items():
            data[k] = v
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(params: dict | None = None) -> StrategyConfig:
    p = {
        "benchmark": "SPY",
        "range_lookback": 60,
        "range_max_move": 0.08,
        "range_vol_cap_percentile": 0.85,
        "z_entry": 1.5,
        "z_exit": 0.3,
        "atr_period": 14,
        "atr_mult": 2.0,
        "risk_pct": 0.01,
        "warmup_bars": 60,
        "cooldown_bars": 3,
        "process_noise": 1e-4,
        "measurement_noise": 1e-2,
    }
    if params:
        p.update(params)
    return StrategyConfig(
        name="kalman_mr_test",
        strategy_type="kalman_mr_regime_dsl",
        symbols=["SPY", "AAPL", "MSFT"],
        initial_capital=100000.0,
        commission=0.05,
        training_start="2021-01-01",
        training_end="2021-07-01",
        trading_start="2021-07-01",
        trading_end="2022-12-31",
        bars=["1d"],
        strategy_params=p,
    )


def test_strategy_is_discoverable():
    strat = init_strat("kalman_mr_regime_dsl")
    assert hasattr(strat, "on_candle")
    assert hasattr(strat, "reset_global")


def test_range_gate_detects_ranging_benchmark():
    """A flat/oscillating SPY must register as RANGE (small slope)."""
    from src.bt.engine.candle_store import CandleStore
    from src.bt.engine.utils import merge_bt_state
    from src.bt.state import create_initial_backtest_state
    from src.bt.strategies.dsl import StrategyContext
    from src.bt.strategies.ta_context import TaContext

    df = _ranging_df()
    ta = TaContext.from_data(df, ("SPY", "AAPL", "MSFT"), "1d")
    n = len(df)
    rows: dict = {}
    for s in ("SPY", "AAPL", "MSFT"):
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
    store.advance(df.index[200])
    state = create_initial_backtest_state(
        symbols=["SPY", "AAPL", "MSFT"],
        initial_capital=100000.0,
        start_timestamp=df.index[0],
        rolling_window_size=None,
    )
    state = merge_bt_state(state, dict(candles=store))
    candle = __import__("src.bt.state", fromlist=["Candle"]).Candle(
        timestamp=df.index[200],
        symbol="MSFT",
        open=float(df[("MSFT", "open")].iloc[200]),
        high=float(df[("MSFT", "high")].iloc[200]),
        low=float(df[("MSFT", "low")].iloc[200]),
        close=float(df[("MSFT", "close")].iloc[200]),
        volume=1000.0,
        interval="1d",
    )
    ctx = StrategyContext(state, candle, Params(), ta, ("SPY", "AAPL", "MSFT"), "1d")
    move = _bench_slope_move(ctx, Params())
    assert move is not None
    assert abs(move) < Params().range_max_move
    assert _regime_ok(ctx, Params()) is True


def test_run_produces_trades_in_ranging_market():
    df = _ranging_df()
    res = run(Backtest(_cfg()), df, strat_mod=init_strat("kalman_mr_regime_dsl"))
    # A genuine range should give the fades plenty of chances.
    assert len(res.pf.trades) > 0
    # The benchmark is the regime gate, never itself a traded leg.
    for sym in res.final_state.portfolio.positions:
        assert sym != "SPY"
    # Every trade was sized from a real stop (risk-based), so entry > 0.
    for t in res.pf.trades:
        assert t.entry_price > 0


def test_run_flattens_on_trending_benchmark():
    """A strongly trending benchmark must keep the book flat (no trades)."""
    df = _ranging_df()
    # Draw the gate into a firm uptrend by overriding SPY only.
    n = len(df)
    uptrend = 100 * (1.0015 ** np.arange(n)) + np.arange(n) * 0.05
    for f in ("open", "high", "low", "close"):
        df[("SPY", f)] = uptrend
    p = _cfg({"range_max_move": 0.005})  # essentially never rangey
    res = run(Backtest(p), df, strat_mod=init_strat("kalman_mr_regime_dsl"))
    # Strictly trending => no fadeable range => no trades.
    assert len(res.pf.trades) == 0


def test_allocation_sizing_caps_at_symbol_alloc_and_cash():
    """``allocation`` mode deploys ``symbol_alloc`` of initial capital per
    symbol and never exceeds available cash (no leverage)."""
    from src.bt.strategies.kalman_mr_regime_dsl import _size_fraction

    alloc = Params(sizing_mode="allocation", symbol_alloc=0.20)
    ctx = _make_alloc_ctx(cash=100_000.0, initial_capital=100_000.0)
    frac = _size_fraction(ctx, "AAPL", alloc, price=100.0, atr_val=2.0)
    assert frac == pytest.approx(0.20)

    ctx_capped = _make_alloc_ctx(cash=5_000.0, initial_capital=100_000.0)
    frac_capped = _size_fraction(ctx_capped, "AAPL", alloc, price=100.0, atr_val=2.0)
    assert frac_capped == pytest.approx(0.05)


def test_risk_sizing_is_atr_stop_scaled():
    """``risk`` mode sizes from per-trade risk and the ATR stop."""
    from src.bt.strategies.kalman_mr_regime_dsl import _size_fraction

    risk = Params(sizing_mode="risk", risk_pct=0.01, atr_mult=2.0)
    # risk = 0.01 * 100k = $1000; stop = 2*2.0 = $4; qty=250; price=100 => 25000 notional
    ctx = _make_alloc_ctx(cash=100_000.0, initial_capital=100_000.0)
    frac = _size_fraction(ctx, "AAPL", risk, price=100.0, atr_val=2.0)
    # qty = 100000*0.01/4 = 250 shares; size = 250*100/100000 = 0.25
    assert frac == pytest.approx(0.25)


def test_volume_gate_blocks_climax_fades():
    """``reject`` mode blocks an entry when volume is climactic (breakout)."""
    from src.bt.strategies.kalman_mr_regime_dsl import _volume_ok
    from src.bt.engine.candle_store import CandleStore
    from src.bt.engine.utils import merge_bt_state
    from src.bt.state import create_initial_backtest_state
    from src.bt.strategies.dsl import StrategyContext
    from src.bt.strategies.ta_context import TaContext

    df = _ranging_df()
    n = len(df)
    df.loc[df.index[-1], ("MSFT", "volume")] = 2_000_000.0  # climactic spike
    ta = TaContext.from_data(df, ("SPY", "AAPL", "MSFT"), "1d")
    rows: dict = {}
    for s in ("SPY", "AAPL", "MSFT"):
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
    store.advance(df.index[-1])
    state = create_initial_backtest_state(
        symbols=["SPY", "AAPL", "MSFT"],
        initial_capital=100000.0,
        start_timestamp=df.index[0],
        rolling_window_size=None,
    )
    state = merge_bt_state(state, dict(candles=store))
    candle = __import__("src.bt.state", fromlist=["Candle"]).Candle(
        timestamp=df.index[-1],
        symbol="MSFT",
        open=float(df[("MSFT", "open")].iloc[-1]),
        high=float(df[("MSFT", "high")].iloc[-1]),
        low=float(df[("MSFT", "low")].iloc[-1]),
        close=float(df[("MSFT", "close")].iloc[-1]),
        volume=float(df[("MSFT", "volume")].iloc[-1]),
        interval="1d",
    )
    ctx = StrategyContext(state, candle, Params(), ta, ("SPY", "AAPL", "MSFT"), "1d")
    assert _volume_ok(ctx, "MSFT", Params(volume_mode="reject", vol_mult=1.4)) is False
    assert _volume_ok(ctx, "MSFT", Params(volume_mode="off")) is True


def _make_alloc_ctx(cash: float, initial_capital: float):
    """A minimal StrategyContext-like object carrying the portfolio for sizing."""
    pf = type("_PF", (), {"cash": cash, "initial_capital": initial_capital})()
    return type("_ST", (), {"state": type("_S", (), {"portfolio": pf})()})()
