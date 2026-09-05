"""ae_bias regime-side-selector tests for cross_momentum_reversal.

Focus (no 88-symbol full feed or DB needed — synthetic OHLCV over a tiny
panel driving the real engine via ``run``):

* ``Params.ae_bias`` defaults OFF, so resolving an ungated config leaves the
  strategy exactly at the baseline (off-path never feeds AE, never touches
  the dual-leg book).
* ``_ae_side_for_trend`` (pure) maps the raw QQQ AE trend (-1/0/1) to a ONE
  side: +1 -> long, -1 -> short (only when ``use_short``), 0 -> no side.
* end-to-end (engine, synthetic QQQ + 4 members): a rising QQQ locking AE
  trend == +1 opens a LONG-only book on the worst-residual names; a falling
  QQQ locking AE trend == -1 opens a SHORT-only book on the top-residual names.

These exercise the new regime-bias mode without a long multi-symbol real run.
The ``ae_bias=False`` no-op equivalence vs the released behavior is asserted
by the real-engine A/B (baseline reproduce) in the acceptance evidence.
"""

import numpy as np
import pandas as pd
from collections import Counter

from src.bt.strategies import init_strat
from src.bt.strategies.cross_momentum_reversal_dsl import (
    STRATEGY_TYPE,
    Params,
    _ae_side_for_trend,
)
from src.bt.engine.backtest import Backtest, run
from src.bt.types import StrategyConfig

_MEMBERS = ["LOWER", "AMID", "ANOLE", "BHIGH"]  # worst -> ... -> best residual


# ---------------------------------------------------------------------------
# synthetic panel helpers
# ---------------------------------------------------------------------------


def _bull_bear(slope: float, n: int = 330, seed: int = 3) -> pd.DataFrame:
    """Synthetic daily OHLCV: QQQ ``slope`` (locked trend) + 4 separable members."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    q = 100 * np.exp(np.cumsum(rng.normal(slope, 0.0035, n)))  # strong drift

    # Member paths clearly separated so residual worst/best is unambiguous.
    paths = {
        "LOWER": np.linspace(80, 35, n),  # hardest faller  -> worst residual
        "AMID": np.linspace(65, 52, n),
        "ANOLE": np.linspace(55, 60, n),
        "BHIGH": np.linspace(40, 80, n),  # strongest riser -> best residual
    }
    cols: dict = {("QQQ", "close"): q}
    for pn in _MEMBERS:
        cols[(pn, "close")] = paths[pn]
    for sym in ["QQQ", *_MEMBERS]:
        c = cols[(sym, "close")].astype(float)
        cols[(sym, "open")] = c * (1 - 0.0002)
        cols[(sym, "high")] = c * 1.001
        cols[(sym, "low")] = c * 0.999
        cols[(sym, "volume")] = np.full(n, 1500.0)
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(use_short: bool) -> StrategyConfig:
    d = {
        "lookback": 21,
        "hold_days": 40,
        "ols_window": 63,
        "benchmark": "QQQ",
        "tail_n": 2,
        "long_share": 0.5,
        "short_share": 0.5,
        "use_short": use_short,
        "warmup_bars": 84,
        "min_total_daily_history": 130,
        "ae_bias": True,
        "entropy_lookback": 25,
        "entropy_num_bins": 10,
    }
    return StrategyConfig(
        name="xmr-ae-test",
        strategy_type=STRATEGY_TYPE,
        symbols=["LOWER", "AMID", "ANOLE", "BHIGH", "QQQ"],
        initial_capital=100000.0,
        commission=0.0,
        training_start="2019-01-01",
        training_end="2019-12-31",
        trading_start="2020-01-01",
        trading_end="2026-01-01",
        bars=["1d"],
        strategy_params=d,
    )


def _side_counts(res) -> tuple[Counter, Counter]:
    from src.bt.state import ActionType

    longs: Counter = Counter()
    shorts: Counter = Counter()
    for t in res.pf.trades:
        row = longs if t.position == ActionType.long else shorts
        row[t.symbol] += 1
    return longs, shorts


# ---------------------------------------------------------------------------
# unit: Params default & pure side map
# ---------------------------------------------------------------------------


def test_ae_bias_defaults_off():
    p = Params()
    assert p.ae_bias is False
    assert p.entropy_lookback == 25
    assert p.entropy_num_bins == 10


def test_ae_side_map_trend_pos_is_long():
    assert _ae_side_for_trend(1, Params(use_short=True)) == "long"
    assert _ae_side_for_trend(1, Params(use_short=False)) == "long"


def test_ae_side_map_trend_neg_is_short_only_when_permitted():
    assert _ae_side_for_trend(-1, Params(use_short=True)) == "short"
    assert _ae_side_for_trend(-1, Params(use_short=False)) is None


def test_ae_side_map_trend_zero_skips():
    assert _ae_side_for_trend(0, Params(use_short=True)) is None
    assert _ae_side_for_trend(0, Params(use_short=False)) is None


# ---------------------------------------------------------------------------
# end-to-end: QQQ AE regime picks the single deployed side
# ---------------------------------------------------------------------------


def test_ae_bias_long_only_in_bull_regime():
    # Rising QQQ locks the AE trend to +1, so the book is LONG-only on the two
    # worst-residual names (LOWER, AMID). No shorts are ever opened.
    df = _bull_bear(slope=+0.003)
    res = run(Backtest(_cfg(use_short=True)), df, strat_mod=init_strat(STRATEGY_TYPE))
    longs, shorts = _side_counts(res)
    assert len(res.pf.trades) > 0
    assert set(longs) >= {"LOWER", "AMID"}, (
        f"expected worst-residual longs, got {dict(longs)}"
    )
    assert len(shorts) == 0, f"bull regime must not short, got {dict(shorts)}"


def test_ae_bias_short_only_in_bear_regime():
    # Falling QQQ locks the AE trend to -1; with use_short it deploys a
    # SHORT-only book on the top-residual names (ANOLE, BHIGH). No longs.
    df = _bull_bear(slope=-0.003)
    res = run(Backtest(_cfg(use_short=True)), df, strat_mod=init_strat(STRATEGY_TYPE))
    longs, shorts = _side_counts(res)
    assert len(res.pf.trades) > 0
    assert set(shorts) >= {"ANOLE", "BHIGH"}, (
        f"expected top-residual shorts, got {dict(shorts)}"
    )
    assert len(longs) == 0, f"bear regime must not go long, got {dict(longs)}"


def test_ae_bias_no_short_when_use_short_off_even_in_bear():
    # Even when the QQQ AE trend is -1, if shorting is globally disallowed the
    # mode must not invent a short leg (side map returns None -> skip).
    df = _bull_bear(slope=-0.003)
    res = run(Backtest(_cfg(use_short=False)), df, strat_mod=init_strat(STRATEGY_TYPE))
    longs, shorts = _side_counts(res)
    assert len(shorts) == 0, "use_short=False must never open a short"
    assert len(longs) == 0, "bear regime with no short offered no long side"
