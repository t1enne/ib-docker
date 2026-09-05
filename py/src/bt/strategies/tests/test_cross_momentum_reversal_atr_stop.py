"""ATR trailing-stop downside tests for cross_momentum_reversal.

``atr_stop`` puts a chandelier ATR trailing stop on every open LONG so a
mean-reversion holding that keeps falling is cut early instead of riding dead
weight to the next monthly rotation. Coverage:

* ``Params.atr_stop`` defaults OFF (byte-identical legacy behaviour).
* pure trail level: ``_atr_trail_level`` = highestHigh - mult*ATR.
* end-to-end: a member that stays in the book while trending DOWN (the worst
  loser — exactly what rotation would keep riding) is force-closed by the ATR
  trail under ``atr_stop=True`` and not under the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.state import ActionType
from src.bt.strategies import init_strat
from src.bt.strategies.cross_momentum_reversal_dsl import (
    STRATEGY_TYPE,
    Params,
)
from src.bt.engine.backtest import Backtest, run
from src.bt.types import StrategyConfig

_MEMBERS = ["LOWER", "AMID", "ANOLE", "BHIGH"]  # worst -> ... -> best residual


def _crash_panel(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Benchmark drifts gently; the two tops-ranked losers keep falling hard.

    LOWER and AMID are the deepest recent losers (long book), and unlike a
    reversal they keep sliding — so holding them to the (long) refresh horizon
    rides a lot of drawdown before any rotation sells them.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    q = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.004, n)))  # gentle bull bench

    t = np.arange(n)
    paths = {
        "LOWER": 100 * np.exp(-0.006 * t) + 2.0,  # deepest monotone faller
        "AMID": 95 * np.exp(-0.0045 * t) + 3.0,  # second-deepest faller
        "ANOLE": np.linspace(60, 58, n) + 3.0 * np.sin(t / 20),  # flat chop
        "BHIGH": np.linspace(40, 95, n),  # steady riser -> top residual
    }
    cols: dict = {("QQQ", "close"): q}
    for pn in _MEMBERS:
        cols[(pn, "close")] = paths[pn]
    for sym in ["QQQ", *_MEMBERS]:
        c = cols[(sym, "close")].astype(float)
        cols[(sym, "open")] = c * (1 - 0.0002)
        cols[(sym, "high")] = c * 1.003
        cols[(sym, "low")] = c * 0.997
        cols[(sym, "volume")] = np.full(n, 1500.0)
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(atr_stop: bool, **overrides) -> StrategyConfig:
    d = {
        "lookback": 21,
        "hold_days": 60,  # long horizon so an unchecked loser bleeds a lot
        "ols_window": 63,
        "benchmark": "QQQ",
        "tail_n": 2,
        "long_share": 0.6,
        "short_share": 0.0,
        "use_short": False,
        "warmup_bars": 84,
        "min_total_daily_history": 130,
        "atr_stop": atr_stop,
        "atr_period": 14,
        "trail_lookback": 10,
        "trail_atr_mult": 2.5,
        "atr_grace_bars": 2,
    }
    d.update(overrides)
    return StrategyConfig(
        name="xmr-atr-test",
        strategy_type=STRATEGY_TYPE,
        symbols=[*_MEMBERS, "QQQ"],
        initial_capital=100000.0,
        commission=0.0,
        training_start="2019-01-01",
        training_end="2019-12-31",
        trading_start="2020-01-01",
        trading_end="2026-01-01",
        bars=["1d"],
        strategy_params=d,
    )


def _atr_close_reasons(res) -> list:
    out = []
    for tr in res.pf.trades:
        if tr.symbol != "LOWER":
            continue
        cr = tr.close_reason
        out.append(str(cr))
    return out


def _long_lower_trades(res) -> list:
    out = []
    for tr in res.pf.trades:
        if tr.symbol == "LOWER" and tr.position == ActionType.long:
            out.append(tr)
    return out


# ---------------------------------------------------------------------------
# unit: defaults + pure trail
# ---------------------------------------------------------------------------


def test_atr_stop_defaults_off():
    p = Params()
    assert p.atr_stop is False
    assert p.atr_period == 14
    assert p.trail_lookback == 10
    assert p.trail_atr_mult == 3.0
    assert p.atr_grace_bars == 3


def test_atr_stop_param_resolution_default_off():
    # A config that omits atr_stop leaves the param off (byte-identical to the
    # ungated legacy behaviour). ``Params.from_dict`` mirrors resolve_params.
    ps = Params.from_dict({})
    assert ps.atr_stop is False


def test_atr_trail_from_math():
    # ``trail = anchor - mult * atr``; degenerate ATR / anchor -> NaN (no stop).
    from src.bt.strategies.cross_momentum_reversal_dsl import _trail_level_from
    import math

    assert abs(_trail_level_from(3.0, 100.0, 6.0) - (100.0 - 18.0)) < 1e-9
    assert math.isnan(_trail_level_from(0.0, 100.0, 6.0))  # zero ATR
    assert math.isnan(_trail_level_from(float("nan"), 100.0, 6.0))


# ---------------------------------------------------------------------------
# end-to-end: a falling long is cut under the ATR stop
# ---------------------------------------------------------------------------


def test_atr_stop_cuts_relentless_loser():
    df = _crash_panel()
    base = run(Backtest(_cfg(False)), df, strat_mod=init_strat(STRATEGY_TYPE))
    guarded = run(Backtest(_cfg(True)), df, strat_mod=init_strat(STRATEGY_TYPE))

    base_reasons = _atr_close_reasons(base)
    guarded_reasons = _atr_close_reasons(guarded)
    assert any("ATR trail stop" in r for r in guarded_reasons), (
        f"expected an ATR-trail exit for the falling long, got {guarded_reasons}"
    )
    assert not any("ATR trail stop" in r for r in base_reasons), (
        "baseline (atr_stop off) must contain no ATR-trail exits"
    )


def test_atr_stop_caps_loser_drawdown():
    # The worst single-leg loss on the crashing long is materially smaller when
    # the ATR stop guards it, versus riding the full (long) hold.
    df = _crash_panel()
    base = run(Backtest(_cfg(False)), df, strat_mod=init_strat(STRATEGY_TYPE))
    guarded = run(Backtest(_cfg(True)), df, strat_mod=init_strat(STRATEGY_TYPE))

    worst_base = min((t.pnl for t in _long_lower_trades(base)), default=0.0)
    worst_guarded = min((t.pnl for t in _long_lower_trades(guarded)), default=0.0)
    # The ATR trail must exit before the bottom: its worst leg is strictly
    # better (less negative) than riding the relentless fall to rotation.
    assert worst_guarded > worst_base, (
        f"ATR stop should cap the crash: guarded worst leg {worst_guarded:.0f} "
        f"vs baseline {worst_base:.0f}"
    )
