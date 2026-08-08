"""A/B reconciliation: momentum screen vs the momentum_regime backtest.

The screen (EMA cross + EMA trend) must agree with the backtest strategy's
*entry direction* (SMA cross + SMA trend) on the same symbols/window over
history. EMA reacts a few bars earlier than SMA on the same cycle, so the test
allows a small time window: every backtest long/short entry must have a
matching-direction screen signal nearby, and every directional screen signal
must land near a matching backtest entry. A screen firing a direction where
the strategy was flat/opposite is a look-ahead or logic-bleed symptom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.bt.strategies.momentum_regime as strat
from src.bt.types import StrategyConfig
from src.bt.engine.backtest import run, Backtest
from src.bt.screen import screen_over_history


def _ohlcv_df(closes: pd.Series, symbols: list[str]) -> pd.DataFrame:
    """Build the (symbol, field) MultiIndex-col DataFrame the engine expects."""
    idx = closes.index
    n = len(idx)
    frames = {}
    for s in symbols:
        frames[s] = {
            "open": closes.to_numpy() * 0.999,
            "high": closes.to_numpy() * 1.002,
            "low": closes.to_numpy() * 0.998,
            "close": closes.to_numpy(),
            "volume": np.full(n, 1_000_000.0),
        }

    cols = pd.MultiIndex.from_product(
        [symbols, ["open", "high", "low", "close", "volume"]]
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for s in symbols:
        for f in ["open", "high", "low", "close", "volume"]:
            df[(s, f)] = frames[s][f]
    df.columns.names = ["symbol", "field"]
    return df


def _engineered_paths() -> dict[str, pd.Series]:
    """Price path engineered to trigger both a long and a short entry.

    The backtest reference (``momentum_regime``) uses SMA cross + SMA trend
    gate; the screen uses EMA cross + EMA trend gate. EMA reacts a few bars
    earlier on the same cycle, so the test allows a small time window and
    requires the *direction* to agree within it, not the exact bar:

      up-leg (BULL)   -> sharp V-dip/recovery -> long entry (both)
      down-leg (BEAR) -> sharp spike/drop    -> short entry (both)
    """
    price: list[float] = []
    p = 100.0
    up = np.log(1.5) / 300  # uptrend
    dip = np.log(0.90) / 30  # -10%% over 30 bars
    rally = np.log(1.18) / 20  # +18%% over 20 bars
    down = np.log(0.70) / 250  # downtrend
    spike = np.log(1.15) / 30  # +15%% spike
    drop = np.log(0.82) / 20  # -18%% drop
    endup = np.log(1.20) / 30  # +20%% recovery (closes the short via CU)

    for _ in range(300):
        p *= np.exp(up)
        price.append(p)
    for _ in range(30):
        p *= np.exp(dip)
        price.append(p)
    for _ in range(20):
        p *= np.exp(rally)
        price.append(p)
    for _ in range(250):
        p *= np.exp(down)
        price.append(p)
    for _ in range(30):
        p *= np.exp(spike)
        price.append(p)
    for _ in range(20):
        p *= np.exp(drop)
        price.append(p)
    for _ in range(30):
        p *= np.exp(endup)
        price.append(p)

    idx = pd.date_range("2022-01-01", periods=len(price), freq="D")
    return {"ABC": pd.Series(price, index=idx)}


def _run_backtest(symbols: list[str], closes_per_symbol: dict[str, pd.Series]):
    config = StrategyConfig(
        name="ab_momentum",
        strategy_type="momentum_regime",
        symbols=symbols,
        initial_capital=100_000,
        commission=0.5,
        training_start="2022-01-01",
        training_end="2023-12-31",
        trading_start="2022-01-01",
        trading_end="2023-12-31",
        bars=["1d"],
        strategy_params={
            "fast": 20,
            "slow": 50,
            "warmup_bars": 60,
            "position_size": 0.2,
            "stop_loss": 0.05,
            "take_profit": 0.20,
        },
        model_updater={
            "type": "dual_online",
            "dual_online": {
                "trend_fast": 50,
                "trend_slow": 200,
                "range_threshold_pct": 0.005,
            },
        },
    )
    # Build a single frame from the first symbol; multi-symbol not needed here.
    closes = closes_per_symbol[symbols[0]]
    bt = Backtest(config)
    df = _ohlcv_df(closes, symbols)
    return run(bt, df, strat_mod=strat)


def _backtest_entries(results) -> list[tuple[pd.Timestamp, str]]:
    """Return (entry timestamp, direction) for every opening trade."""
    return [(t.entry_time, t.position.value) for t in results.pf.trades]


def _screen_signal_bars(hist) -> dict[pd.Timestamp, str]:
    """Map every bar where the screen is non-flat to its direction."""
    out: dict[pd.Timestamp, str] = {}
    for ts, res in hist.items():
        nonflat = [r for r in res if r.action != "flat"]
        if nonflat:
            out[ts] = nonflat[0].action
    return out


def test_momentum_screen_matches_backtest_direction():
    paths = _engineered_paths()
    symbols = ["ABC"]
    results = _run_backtest(symbols, paths)

    entries = _backtest_entries(results)
    assert len(entries) > 0, "backtest must produce at least one entry"

    # Run the screen cursor-safe over the same frames.
    frames = (("ABC", _mk_frame_for_screen(paths["ABC"])),)
    hist = screen_over_history(frames, "momentum", {})
    screen_dir = _screen_signal_bars(hist)

    # Tolerance: the screen uses EMA cross, the backtest SMA cross. EMA reacts
    # a few bars earlier on the same cycle, so allow a small window both sides
    # and require the *direction* to agree within it.
    tol = pd.Timedelta(days=15)

    # 1. Every backtest entry has a matching-direction screen signal nearby.
    for ts, d in entries:
        nearby = [sd for sts, sd in screen_dir.items() if abs(sts - ts) <= tol]
        assert d in nearby, (
            f"backtest entry {d}@{ts.date()} has no matching screen signal "
            f"within {tol.days}d (screen signals nearby: {nearby})"
        )

    # 2. Every directional screen signal lands near a matching backtest entry
    #    (no contradiction: a screen long where the strategy was bearish/flat is
    #    a look-ahead or logic bleed symptom).
    for sts, sd in screen_dir.items():
        nearby_dirs = [d for t, d in entries if abs(sts - t) <= tol]
        assert sd in nearby_dirs, (
            f"screen {sd}@{sts.date()} has no matching backtest entry nearby "
            f"(backtest entries nearby: {nearby_dirs})"
        )


def _mk_frame_for_screen(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=closes.index,
    )
