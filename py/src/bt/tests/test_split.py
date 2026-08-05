"""Tests for walk-forward / single-anchor split logic."""

from __future__ import annotations

import pandas as pd
import pytest
import src.bt.split as split_mod

from src.bt.split import (
    FoldMetrics,
    SplitReport,
    anchor_split,
    walk_forward_folds,
    _reset_strategy_state,
)
from src.bt.state import PortfolioResult
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp


def _cfg(
    trading_start: str = "2015-01-02",
    trading_end: str = "2025-12-31",
    training_start: str = "2015-01-01",
) -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="sector_mean_reversion_trail",
        symbols=["XLB", "XLV", "XLY", "XLU", "SPY"],
        stop_loss=0.15,
        take_profit=0.3,
        initial_capital=50000,
        position_size=0.45,
        commission=0.1,
        training_start=training_start,
        training_end=training_start,
        trading_start=trading_start,
        trading_end=trading_end,
        bars=["1d"],
        strategy_params={"momentum_lookback": 30},
        model_params={},
    )


def _fake_result(sharpe: float) -> PortfolioResult:
    import pandas as pd

    eq = pd.Series([1000.0, 1100.0])
    from src.bt.state import Trade, TradeStatus, ActionType

    win = Trade(
        entry_time=parse_timestamp("2020-01-02"),
        entry_price=10.0,
        exit_time=parse_timestamp("2020-02-01"),
        exit_price=11.0,
        last_price=11.0,
        symbol="XLB",
        position=ActionType.long,
        qty=1.0,
        stop_loss=9.0,
        take_profit=12.0,
        pnl=1.0,
        status=TradeStatus.closed,
    )
    return PortfolioResult(
        total_return=0.1,
        sharpe_ratio=sharpe,
        trades=(win,),
        equity_curve=eq,
        annual_return=0.1,
        max_drawdown=-0.05,
        calmar_ratio=2.0,
    )


def test_anchor_split_windows() -> None:
    cfg = _cfg()
    folds = anchor_split(cfg, parse_timestamp("2020-12-31"))
    assert len(folds) == 1
    f = folds[0]
    assert f.is_start == pd.Timestamp("2015-01-02")
    assert f.is_end == pd.Timestamp("2020-12-31")
    assert f.oos_start > f.is_end  # disjoint, adjacent
    assert f.oos_end == pd.Timestamp("2025-12-31")
    # coverage
    assert f.is_start <= f.is_end < f.oos_start <= f.oos_end


def test_anchor_split_respects_train_start() -> None:
    cfg = _cfg()
    folds = anchor_split(
        cfg, parse_timestamp("2020-12-31"), train_start=parse_timestamp("2014-01-01")
    )
    assert folds[0].is_start == pd.Timestamp("2014-01-01")


def test_anchor_split_is_end_past_trading_end_raises() -> None:
    cfg = _cfg(trading_end="2020-12-31")
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2020-12-31"))  # equal to end
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2021-01-01"))  # past end


def test_anchor_split_is_end_before_start_raises() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2015-01-01"))


def test_walk_forward_folds_expand_is() -> None:
    cfg = _cfg()
    folds = walk_forward_folds(cfg, n_folds=3, min_is_years=0.0)
    # exactly three non-empty folds
    assert len(folds) == 3
    # IS expands monotonically: each is_start is trading_start, is_end grows
    for f in folds:
        assert f.is_start == pd.Timestamp("2015-01-02")
    assert folds[0].is_end < folds[1].is_end < folds[2].is_end
    # OOS windows disjoint, adjacent to IS, non-empty
    for f in folds:
        assert f.is_end < f.oos_start <= f.oos_end
    # last fold OOS reaches trading_end
    assert folds[2].oos_end == pd.Timestamp("2025-12-31")
    # windows never overlap
    for i in range(2):
        assert folds[i].oos_end <= folds[i + 1].oos_start


def test_walk_forward_folds_explicit_offset() -> None:
    cfg = _cfg()
    folds = walk_forward_folds(
        cfg, n_folds=4, oos_length=pd.DateOffset(years=2), min_is_years=0.0
    )
    assert len(folds) == 4
    # interior folds: OOS ~= the 2y offset; final fold runs to trading_end
    for f in folds:
        # OOS always non-empty and after IS
        assert f.is_end < f.oos_start <= f.oos_end
    assert folds[-1].oos_end == pd.Timestamp("2025-12-31")
    for f in folds[:-1]:
        assert f.oos_end - f.oos_start <= pd.Timedelta(days=800)


def test_walk_forward_folds_offset_overflow_clamps_and_warns() -> None:
    """An explicit offset that walks past trading_end must clamp, not walk
    out of range; fewer-than-requested folds warn."""
    cfg = _cfg()
    with pytest.warns(UserWarning, match="Only 1 non-empty fold") as record:
        folds = walk_forward_folds(
            cfg, n_folds=5, oos_length=pd.DateOffset(years=6), min_is_years=0.0
        )
    assert len(record) >= 1
    # produced folds stay strictly inside trading range
    assert len(folds) >= 1
    start = pd.Timestamp(cfg.trading_start)
    end = pd.Timestamp(cfg.trading_end)
    for f in folds:
        assert start <= f.is_start <= f.oos_end
        assert f.oos_end <= end
        assert f.is_end < f.oos_start <= f.oos_end


def test_walk_forward_folds_offset_totally_past_end_is_empty() -> None:
    """Offset larger than the whole span => no valid fold, clean empty list
    (+ warning). No degenerate far-future windows."""
    cfg = _cfg()
    with pytest.warns(UserWarning, match="Only 0 non-empty fold"):
        folds = walk_forward_folds(
            cfg, n_folds=5, oos_length=pd.DateOffset(years=500), min_is_years=0.0
        )
    assert folds == []


def test_walk_forward_folds_indices_contiguous_after_skip() -> None:
    """Skipping empty OOS chunks must not leave index gaps."""
    cfg = _cfg()
    with pytest.warns(UserWarning):
        folds = walk_forward_folds(
            cfg, n_folds=5, oos_length=pd.DateOffset(years=6), min_is_years=0.0
        )
    assert [f.index for f in folds] == list(range(len(folds)))


def test_walk_forward_folds_require_positive_folds() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        walk_forward_folds(cfg, n_folds=0)


def test_walk_forward_folds_train_start_must_precede_is() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        walk_forward_folds(cfg, n_folds=2, train_start=parse_timestamp("2020-01-01"))


def _fold(i: int) -> split_mod.TestFold:
    offset = pd.offsets.DateOffset(years=i)
    return split_mod.TestFold(
        index=i,
        is_start=parse_timestamp("2015-01-02"),
        is_end=parse_timestamp("2020-01-01") + offset,
        oos_start=parse_timestamp("2021-01-01") + offset,
        oos_end=parse_timestamp("2022-01-01") + offset,
    )


def test_metrics_projected() -> None:
    folds = (
        FoldMetrics(
            fold=_fold(0), in_sample=_fake_result(2.0), out_of_sample=_fake_result(1.5)
        ),
        FoldMetrics(
            fold=_fold(1), in_sample=_fake_result(2.0), out_of_sample=_fake_result(0.5)
        ),
        FoldMetrics(
            fold=_fold(2), in_sample=_fake_result(2.0), out_of_sample=_fake_result(1.0)
        ),
    )
    report = SplitReport(config_name="t", params={"momentum_lookback": 30}, folds=folds)
    assert report.mean_oos_sharpe() == pytest.approx((1.5 + 0.5 + 1.0) / 3)
    assert report.min_oos_sharpe() == pytest.approx(0.5)
    # oos/is degradation = mean(oos) / mean(is)
    assert report.oos_vs_is_degradation() == pytest.approx(1.0 / 2.0)
    assert list(report.oos_sharpe_series()) == pytest.approx([1.5, 0.5, 1.0])


def test_metrics_empty_folds() -> None:
    report = SplitReport(config_name="t", params={}, folds=())
    assert report.mean_oos_sharpe() == 0.0
    assert report.min_oos_sharpe() == 0.0
    assert report.oos_vs_is_degradation() == 0.0


def test_split_report_to_dict_counts_only_closed_trades() -> None:
    """trade_count must match the closed-trade denominator used by win_rate,
    not include still-open trades."""
    import pandas as pd

    from src.bt.state import ActionType, Trade, TradeStatus

    win = Trade(
        entry_time=parse_timestamp("2020-01-02"),
        entry_price=10.0,
        exit_time=parse_timestamp("2020-02-01"),
        exit_price=11.0,
        last_price=11.0,
        symbol="XLB",
        position=ActionType.long,
        qty=1.0,
        stop_loss=9.0,
        take_profit=12.0,
        pnl=1.0,
        status=TradeStatus.closed,
    )
    open_trade = Trade(
        entry_time=parse_timestamp("2020-01-02"),
        entry_price=10.0,
        exit_time=parse_timestamp("2020-02-01"),
        exit_price=11.0,
        last_price=11.0,
        symbol="XLB",
        position=ActionType.long,
        qty=1.0,
        stop_loss=9.0,
        take_profit=12.0,
        pnl=0.0,
        status=TradeStatus.open,
    )
    eq = pd.Series([1000.0, 1100.0])
    pf = PortfolioResult(
        total_return=0.1,
        sharpe_ratio=1.0,
        trades=(win, open_trade),
        equity_curve=eq,
        annual_return=0.1,
        max_drawdown=-0.05,
        calmar_ratio=2.0,
    )
    report = SplitReport(
        config_name="t",
        params={"momentum_lookback": 30},
        folds=(FoldMetrics(fold=_fold(0), in_sample=pf, out_of_sample=pf),),
    )
    d = split_mod.split_report_to_dict(report)
    fold0 = d["folds"][0]
    # 2 trades total but only 1 closed => trade_count 1, win_rate 1.0
    assert fold0["is"]["trade_count"] == 1
    assert fold0["is"]["win_rate"] == pytest.approx(1.0)
    assert set(fold0["is"]) == {
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "sortino_ratio",
        "win_rate",
        "trade_count",
    }


def test_reset_strategy_state_clears_module_dicts() -> None:
    import src.bt.strategies.sector_mean_reversion_trail as trail_mod

    trail_mod.GLOBAL["cooldown"]["XLB"] = 3
    trail_mod.GLOBAL["trails"]["pos-1"] = 42.0
    _reset_strategy_state(trail_mod)
    assert trail_mod.GLOBAL == {"cooldown": {}, "trails": {}}


def test_reset_strategy_state_rebinds_non_dict_state() -> None:
    """Non-dict state (Timestamp, set, int) must reset too — the old
    heuristic only cleared uppercase dict globals and bled these across folds."""
    import src.bt.strategies.sector_mean_reversion as smr_mod

    smr_mod.GLOBAL["cooldown"]["XLB"] = 3
    smr_mod.GLOBAL["regime_ts"] = parse_timestamp("2020-01-02")
    _reset_strategy_state(smr_mod)
    assert smr_mod.GLOBAL == {
        "cooldown": {},
        "regime_cache": {},
        "regime_ts": None,
        "regime_candles_cache": {},
    }


def test_reset_strategy_state_clears_set_state() -> None:
    import src.bt.strategies.trend_pullback_atr_trail as trail_variant

    trail_variant.GLOBAL["in_trail"].add("pos-1")
    _reset_strategy_state(trail_variant)
    assert trail_variant.GLOBAL["in_trail"] == set()
    assert trail_variant.GLOBAL["cooldowns"] == {}
    assert trail_variant.GLOBAL["weekly_cache"] == {}


def test_reset_strategy_state_noop_without_hook() -> None:
    import types as types_mod

    stateless = types_mod.ModuleType("fake")  # no reset_global attr
    _reset_strategy_state(stateless)  # should not raise
