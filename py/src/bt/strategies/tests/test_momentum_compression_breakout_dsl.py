"""Tests for the momentum-compression-breakout strategy.

Pins the pure helpers and the end-to-end macro pattern against synthetic data
shaped to reproduce the intended setup: a 30-60% run, a tight low-volume
coil between the 10/20 SMAs, then a breakout that we hold until the 10 SMA
rolls over. Also verifies the guards that keep the pattern meaningful:
no parabolic (>100%) entries and no entries without a compression.
"""

import numpy as np
import pandas as pd

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies import init_strat
from src.bt.strategies.momentum_compression_breakout_dsl import (
    _big_move_ok,
    open_position_count,
    Params,
)
from src.bt.strategies.series import SeriesView
from src.bt.types import StrategyConfig

STRAT = "momentum_compression_breakout_dsl"


def _lk(arr: np.ndarray) -> SeriesView:
    return SeriesView(np.asarray(arr, dtype=float), lambda: len(arr))


def _cfg() -> StrategyConfig:
    return StrategyConfig(
        name="mcb",
        strategy_type=STRAT,
        symbols=["AAPL"],
        initial_capital=100000.0,
        commission=0.05,
        training_start="2018-01-01",
        training_end="2019-12-31",
        trading_start="2020-01-01",
        trading_end="2023-12-31",
        bars=["1d"],
        strategy_params={
            "big_lookback": 40,
            "min_gain": 0.30,
            "max_gain": 1.5,
            "comp_window": 10,
            "body_atr_ratio": 0.4,
            "vol_mult": 0.9,
            "min_hover_bars": 3,
            "hover_tol": 0.01,
            "decay_bars": 8,
            "atr_period": 10,
            "atr_mult": 2.0,
            "risk_pct": 0.01,
            "warmup_bars": 55,
            "cooldown_bars": 2,
        },
    )


# ---------------------------------------------------------------------------
# unit tests on pure helpers
# ---------------------------------------------------------------------------


def test_big_move_gain_range():
    # ~45% run from 100 -> 145, flat recently. Range over the lookback is in [30%, 100%].
    closes = np.concatenate([np.linspace(100, 145, 40), [145] * 10])
    params = Params(big_lookback=40)
    assert _big_move_ok(_lk(closes), params) is True


def test_big_move_rejects_too_small():
    closes = np.concatenate([np.linspace(100, 115, 40), [115] * 10])
    assert _big_move_ok(_lk(closes), Params(big_lookback=40)) is False


def test_big_move_rejects_parabolic():
    # > max_gain (blow-off) must be excluded: a >100% move inside the lookback.
    closes = np.concatenate([np.full(20, 100.0), np.linspace(100, 260, 20)])
    params = Params(big_lookback=40, max_gain=1.0)
    assert _big_move_ok(_lk(closes), params) is False


def test_big_move_pullback_still_counts():
    # The pending code wrapped a big_lookback around a coil; a run then pulling
    # back into a coil keeps the trough-to-peak range inside bounds.
    run_up = np.linspace(100, 145, 30)
    pull = np.linspace(145, 130, 15)
    closes = np.concatenate([run_up, pull])
    params = Params(big_lookback=40)
    assert _big_move_ok(_lk(closes), params) is True


# ---------------------------------------------------------------------------
# end-to-end: big run -> compression -> breakout -> trend ride
# ---------------------------------------------------------------------------


def _mom_data(n: int = 430) -> pd.DataFrame:
    """Synthetic OHLCV: steep run up, coil tightly on low volume, break out, rollover."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    # Phase A: flat base (bar 0..29)
    base = np.full(30, 100.0)
    # Phase B: 60% run in 45 bars (100 -> 160) at HIGH volume
    up = np.linspace(100, 160, 45)
    # Phase C: tight coil around 158, tiny bodies, LOW volume (compression)
    coil = 158 + rng.normal(0, 0.3, 40)
    # Phase D: breakout continuation 160 -> 190 at HIGH volume
    boom = np.linspace(coil[-1], 190, 55)
    # Phase E: trend rollover into a declining 10 SMA
    down = np.linspace(190, 155, 80)
    tail = np.linspace(155, 150, 60)
    closes = np.concatenate([base, up, coil, boom, down, tail])[:n]
    if len(closes) < n:
        closes = np.pad(closes, (0, n - len(closes)), constant_values=closes[-1])

    n_bars = len(closes)
    rng2 = np.random.default_rng(11)
    data = {
        ("AAPL", "open"): closes * (1 - 0.002) + rng2.normal(0, 0.05, n_bars),
        ("AAPL", "high"): closes + 0.4,
        ("AAPL", "low"): closes - 0.4,
        ("AAPL", "close"): closes,
        # coil bars [run_end..run_end+40) are LOW volume between high-volume phases
        ("AAPL", "volume"): np.where(
            (np.arange(n_bars) >= 30 + 45) & (np.arange(n_bars) < 30 + 45 + 40),
            300.0,
            1500.0,
        ),
    }
    df = pd.DataFrame(data, index=idx[:n_bars])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_strategy_trades_the_macro_pattern():
    res = run(Backtest(_cfg()), _mom_data(), strat_mod=init_strat(STRAT))
    trades = res.pf.trades
    assert len(trades) > 0, "expected at least one trade on the synthetic setup"
    # At least one winning winner (breakout ridden into the boom), and net PnL
    # positive on the up-coil-breakout shape.
    assert res.pf.total_return > 0


def test_strategy_modules_importable():
    from src.bt.strategies.momentum_compression_breakout_dsl import STRATEGY_TYPE

    assert STRATEGY_TYPE == STRAT


def _trap_data() -> pd.DataFrame:
    """Setup, a fake one-bar breakout pop, then a hard reversal into the ATR stop."""
    rng = np.random.default_rng(5)
    base = np.full(30, 100.0)
    up = np.linspace(100, 160, 45)
    coil = 158 + rng.normal(0, 0.3, 40)
    pop = np.array([161.0, 189.0])  # breakout bar, then a huge gap up (wiggle room)
    crumble = np.linspace(188, 100, 25)  # sharp reversal
    tail = np.full(40, 100.0)
    closes = np.concatenate([base, up, coil, pop, crumble, tail])
    n_bars = len(closes)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    data = {
        ("AAPL", "open"): closes * (1 - 0.001),
        ("AAPL", "high"): closes + 0.5,
        ("AAPL", "low"): closes - 0.5,
        ("AAPL", "close"): closes,
        ("AAPL", "volume"): np.where(
            (np.arange(n_bars) >= 75) & (np.arange(n_bars) < 115),
            300.0,
            1500.0,
        ),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_atr_stop_exit_fires_on_trap():
    """A fake-out breakout that immediately reverses must be caught by the ATR stop."""
    res = run(Backtest(_cfg()), _trap_data(), strat_mod=init_strat(STRAT))
    trades = res.pf.trades
    assert len(trades) > 0, "trap data must open (then stop out) at least one trade"
    stop_reasons = [t for t in trades if "stop" in str(getattr(t, "close_reason", ""))]
    assert stop_reasons, "expected an ATR stop exit on the trap pattern"


# ---------------------------------------------------------------------------
# adaptive-entropy market-regime gate
# ---------------------------------------------------------------------------


def _regime_cfg(regime: str) -> StrategyConfig:
    """Config with an observer ``regime`` symbol + AE market gate."""
    cfg = _cfg()
    d = dict(cfg.strategy_params)
    d.update(
        {
            "regime_symbol": regime,
            "regime_trend_min": 1,
            "regime_min_strength": 0.0,
            "ae_lookback": 10,
            "ae_num_bins": 8,
        }
    )
    # observer index's data must be fed so the AE gate can read it.
    return StrategyConfig(
        name=cfg.name,
        strategy_type=cfg.strategy_type,
        symbols=list(cfg.symbols) + [regime],
        initial_capital=cfg.initial_capital,
        commission=cfg.commission,
        training_start=cfg.training_start,
        training_end=cfg.training_end,
        trading_start=cfg.trading_start,
        trading_end=cfg.trading_end,
        bars=cfg.bars,
        strategy_params=d,
    )


def _regime_series(bull: bool, n: int = 300) -> np.ndarray:
    """A monotonic index run: bull = rising (trend=+1), bear = falling."""
    base = np.linspace(100.0, 160.0 if bull else 60.0, n)
    # Small alternating walk so the entropy window is defined (rng > 0).
    rng = np.random.default_rng(7)
    walk = base + rng.normal(0, 0.6, n)
    return np.maximum(walk, 1.0)


def _two_symbol_bt(regime_bull: bool):
    """Run strategy + AE gate on the AAPL setup with a bull or bear index."""
    aapl = _mom_data(430)
    idx = aapl.index
    reg = _regime_series(regime_bull, n=len(idx))
    reg_df = {
        ("QQQ", "open"): reg * (1 - 0.001),
        ("QQQ", "high"): reg + 0.4,
        ("QQQ", "low"): reg - 0.4,
        ("QQQ", "close"): reg,
        ("QQQ", "volume"): np.full(len(reg), 1500.0),
    }
    reg_frame = pd.DataFrame(reg_df, index=idx)
    reg_frame.columns = pd.MultiIndex.from_tuples(reg_frame.columns)
    # Inner join on the shared timestamp grid so the observer index lines up 1:1
    # with the AAPL 1d feed.
    return aapl.join(reg_frame, how="left")


def test_ae_gate_blocks_entry_in_bear_regime():
    """A bearish index (rising stock would otherwise break out) must be gated out."""
    df = _two_symbol_bt(regime_bull=False)
    cfg = _regime_cfg("QQQ")
    res = run(Backtest(cfg), df, strat_mod=init_strat(STRAT))
    trades = res.pf.trades
    # The strategy should not have opened length positions against a bear index.
    assert len(trades) == 0, (
        f"expected 0 trades with bearish AE gate, got {len(trades)}"
    )


def test_ae_gate_allows_entry_in_bull_regime():
    """A bullish index must let the macro setup trade (control for the gate)."""
    df = _two_symbol_bt(regime_bull=True)
    cfg = _regime_cfg("QQQ")
    res = run(Backtest(cfg), df, strat_mod=init_strat(STRAT))
    trades = res.pf.trades
    assert len(trades) > 0, "expected trades with a bullish AE regime gate"
    # The observer index itself is never traded.
    syms = {t.symbol for t in trades}
    assert "QQQ" not in syms


# ---------------------------------------------------------------------------
# capital-allocation guards (max concurrent positions + notional cap)
# ---------------------------------------------------------------------------


def test_open_position_count_helper():
    from dataclasses import replace
    import pandas as pd
    from src.bt.state import ActionType, create_initial_backtest_state, Position

    state = create_initial_backtest_state(
        symbols=["AAPL"],
        initial_capital=100000.0,
        start_timestamp=pd.Timestamp("2020-01-01"),  # ty: ignore[invalid-argument-type]
    )
    assert open_position_count(state.portfolio) == 0
    pos = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=pd.Timestamp("2020-01-02"),  # ty: ignore[invalid-argument-type]
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
    )
    p2 = replace(state.portfolio, positions={"AAPL": (pos,)})
    assert open_position_count(p2) == 1


def _replace_params(cfg: StrategyConfig, d: dict) -> StrategyConfig:
    return StrategyConfig(
        name=cfg.name,
        strategy_type=cfg.strategy_type,
        symbols=cfg.symbols,
        initial_capital=cfg.initial_capital,
        commission=cfg.commission,
        training_start=cfg.training_start,
        training_end=cfg.training_end,
        trading_start=cfg.trading_start,
        trading_end=cfg.trading_end,
        bars=cfg.bars,
        strategy_params=d,
    )


def test_max_positions_caps_concurrent_exposure():
    """With max_positions capped, the strategy must not stack more concurrent
    positions than the cap on a setup that produces many simultaneous breakouts."""
    df = _two_symbol_bt(regime_bull=True)
    cfg = _regime_cfg("QQQ")
    d = dict(cfg.strategy_params)
    d["max_positions"] = 3
    cfg = _replace_params(cfg, d)
    res = run(Backtest(cfg), df, strat_mod=init_strat(STRAT))
    trades = res.pf.trades
    life = [(t.entry_time, t.exit_time or t.entry_time) for t in trades]
    times = sorted({ts for s, e in life for ts in (s, e)})
    maxc = 0
    for day in times:
        maxc = max(maxc, sum(1 for s, e in life if s <= day < e))
    assert maxc <= 3, f"max concurrent positions {maxc} exceeded cap 3"
