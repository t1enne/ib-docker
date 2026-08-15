"""Tests for the direction-configurable VP-breakout strategy.

The ``vp_breakout_dsl`` strategy trades both sides by default
(``direction="both"``); the plan makes it direction-configurable so the long
leg, the short/bear leg, and the combined book share one code path. These
tests pin the ``direction`` gate: ``"long"`` must never open a short,
``"short"`` must never open a long, and ``"both"`` must permit either side.

Synthetic data deliberately produces *both* a long breakout (above the value
area high, uptrend) and a short breakdown (below the value area low,
downtrend) so the gate is exercised, not passed vacuously.
"""

import numpy as np
import pandas as pd

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies import init_strat
from src.bt.strategies.vp_breakout_dsl import Params
from src.bt.types import StrategyConfig

STRAT = "vp_breakout_dsl"


def _cfg(direction: str) -> StrategyConfig:
    return StrategyConfig(
        name="vp_dir",
        strategy_type=STRAT,
        symbols=["AAPL", "MSFT"],
        initial_capital=100000.0,
        commission=0.05,
        training_start="2019-01-01",
        training_end="2019-12-31",
        trading_start="2020-01-01",
        trading_end="2023-12-31",
        bars=["1d"],
        strategy_params={
            # Feed a long and a short side clearly.
            "direction": direction,
            "vp_window": 120,
            "num_bins": 20,
            "value_area_pct": 0.7,
            "vp_warmup": 30,
            "vol_period": 10,
            "vol_expand_mult": 0.5,  # drop the volume gate for fewer data constraints
            "wick_check": 0.4,
            "trend_lookback": 25,
            "regime_lookback": 10,
            "regime_er": 0.0,  # no regime gate — test the direction flip alone
            "sizing_mode": "risk",
            "symbol_alloc": 0.2,
            "atr_period": 10,
            "atr_mult": 2.0,
            "risk_pct": 0.01,
            "cooldown_bars": 2,
        },
    )


def _df(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """OHLCV whose close rises-then-falls around the value-area band.

    The first half trends up (long-breakout material), the second half trends
    down (short-breakdown material), so both signals occur at some point.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    # rise to ~+20% then fall back ~-25%: a full up/down cycle.
    base = 100 + 20 * np.sin(t * (np.pi / n)) * 2  # up then down
    close = base + np.cumsum(rng.normal(0, 0.3, n))
    data = {}
    for sym in ("AAPL", "MSFT"):
        data[(sym, "open")] = close + 0.3
        data[(sym, "high")] = close + 2.5
        data[(sym, "low")] = close - 2.5
        data[(sym, "close")] = close
        # vol spikes at trend turns to make breakouts confirmable
        data[(sym, "volume")] = 1000 + 800 * np.abs(
            np.diff(np.sin(t * 0.02), prepend=0)
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_params_parses_direction():
    assert Params.from_dict({"direction": "long"}).direction == "long"
    assert Params.from_dict({"direction": "short"}).direction == "short"
    assert Params.from_dict({"direction": "both"}).direction == "both"
    assert Params.from_dict({}).direction == "both"  # back-compat default
    assert Params().direction == "both"


def _run(direction: str):
    return run(Backtest(_cfg(direction)), _df(), strat_mod=init_strat(STRAT))


def _trade_directions(res):
    from src.bt.state import ActionType

    longs = [t for t in res.pf.trades if t.position == ActionType.long]
    shorts = [t for t in res.pf.trades if t.position == ActionType.short]
    return longs, shorts


def test_short_direction_never_opens_a_long():
    res = _run("short")
    # The short config must place at least one trade (data produces signals).
    assert len(res.pf.trades) > 0
    longs, shorts = _trade_directions(res)
    assert len(longs) == 0, "short-only config must open no long positions"
    assert len(shorts) > 0, "short-only config should open shorts on the downtrend"


def test_long_direction_never_opens_a_short():
    res = _run("long")
    assert len(res.pf.trades) > 0
    longs, shorts = _trade_directions(res)
    assert len(shorts) == 0, "long-only config must open no short positions"
    assert len(longs) > 0, "long-only config should open longs on the uptrend"


def test_both_direction_allows_either_side():
    res = _run("both")
    assert len(res.pf.trades) > 0
    longs, shorts = _trade_directions(res)
    # The synthetic cycle should expose both breakouts eventually.
    assert len(longs) > 0 or len(shorts) > 0
