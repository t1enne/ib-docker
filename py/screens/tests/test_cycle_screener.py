"""Tests for the cycle_screener module.

Tests each Layer pure function independently with small DataFrames
and hand-crafted prices to verify known outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from screens.cycle_screener import (
    _returns,
    _relative_strength,
    _leadership_ranking,
    _calc_momentum_scores,
    _compute_ratios,
    _ratio_trends,
    _calc_breadth,
    _calc_credit_scores,
    _calc_rates_scores,
    _score_risk,
    _score_growth,
    _score_inflation,
    _score_breadth,
    _score_liquidity,
    _classify_regime,
    _cross_asset_confirmation,
    CycleScreen,
    ALL_ASSETS,
    _BENCHMARK,
)


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def simple_closes() -> dict[str, pd.Series]:
    """Simple close series for all assets — prices rise linearly for most."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.linspace(100, 150, n)  # gentle uptrend

    closes: dict[str, pd.Series] = {}
    for ticker in ALL_ASSETS:
        noise = np.random.default_rng(42).normal(0, 2, n)
        prices = base + noise
        closes[ticker] = pd.Series(prices, index=dates)

    return closes


@pytest.fixture
def equity_only_closes() -> dict[str, pd.Series]:
    """Minimal fixture with just SPY and a few sectors."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.linspace(100, 150, n)

    closes: dict[str, pd.Series] = {}
    for ticker in ["SPY", "XLF", "XLK", "XLI", "XLV", "XLP"]:
        noise = np.random.default_rng(42).normal(0, 2, n)
        closes[ticker] = pd.Series(base + noise, index=dates)
    return closes


@pytest.fixture
def flat_closes() -> dict[str, pd.Series]:
    """All prices flat at 100 — zero momentum."""
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    closes: dict[str, pd.Series] = {}
    for ticker in ALL_ASSETS:
        closes[ticker] = pd.Series(np.full(n, 100.0), index=dates)
    return closes


# ── Test helper: _returns ────────────────────────────────────


class TestReturns:
    def test_positive_return(self) -> None:
        s = pd.Series([100.0, 110.0, 120.0])
        assert _returns(s, 2) == pytest.approx(0.2)  # 120/100 - 1

    def test_negative_return(self) -> None:
        s = pd.Series([100.0, 90.0, 80.0])
        assert _returns(s, 2) == pytest.approx(-0.2)

    def test_insufficient_data(self) -> None:
        s = pd.Series([100.0])
        assert _returns(s, 10) == 0.0

    def test_exact_window(self) -> None:
        s = pd.Series([100.0, 105.0])
        assert _returns(s, 1) == pytest.approx(0.05)

    def test_zero_denominator(self) -> None:
        s = pd.Series([0.0, 50.0, 100.0])
        assert _returns(s, 2) == 0.0  # would be inf, but we guard


# ── Test layer 1: Relative strength ────────────────────────────


class TestRelativeStrength:
    def test_simple_rs(self) -> None:
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        bench_close = pd.Series(np.linspace(100, 110, n), index=dates)
        closes: dict[str, pd.Series] = {
            "XLF": pd.Series(np.linspace(100, 120, n), index=dates),
            "XLP": pd.Series(np.linspace(100, 105, n), index=dates),
        }

        rs = _relative_strength(closes, bench_close, window=63)
        # XLF outperformed SPY: 20% vs 10% = 10pp
        # XLP underperformed: 5% vs 10% = -5pp
        assert "XLF" in rs
        assert "XLP" in rs
        # XLF should have higher RS than XLP
        assert rs["XLF"] > rs["XLP"]

    def test_empty_closes(self) -> None:
        rs = _relative_strength({}, pd.Series([100.0]))
        assert rs == {}


# ── Test layer 2: Leadership ranking ───────────────────────────


class TestLeadershipRanking:
    def test_ranking_order(self) -> None:
        rs = {"XLK": 0.05, "XLF": 0.10, "XLV": -0.02, "XLP": 0.08}
        ranked = _leadership_ranking(rs)
        assert len(ranked) == 4
        assert ranked[0][0] == "XLF"  # highest RS
        assert ranked[-1][0] == "XLV"  # lowest RS
        # Verify descending order
        scores = [s for _, s in ranked]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_empty_input(self) -> None:
        assert _leadership_ranking({}) == []


# ── Test layer 3: Momentum scores ──────────────────────────────


class TestMomentumScores:
    def test_uptrend_momentum(self) -> None:
        n = 250
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        series = pd.Series(np.linspace(100, 150, n), index=dates)
        closes = {"SPY": series}
        scores = _calc_momentum_scores(closes)
        assert "SPY" in scores
        assert scores["SPY"] > 50.0  # Trending up -> above neutral

    def test_downtrend_momentum(self) -> None:
        n = 250
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        series = pd.Series(np.linspace(150, 100, n), index=dates)
        closes = {"SPY": series}
        scores = _calc_momentum_scores(closes)
        assert scores["SPY"] < 50.0  # Trending down -> below neutral

    def test_flat_momentum(self, flat_closes: dict[str, pd.Series]) -> None:
        scores = _calc_momentum_scores(flat_closes)
        for val in scores.values():
            assert val == pytest.approx(50.0, abs=1.0)  # flat ~ neutral

    def test_insufficient_data(self) -> None:
        s = pd.Series([100.0] * 50)  # too short
        scores = _calc_momentum_scores({"SPY": s})
        assert scores["SPY"] == 0.0


# ── Test layer 4: Ratios ───────────────────────────────────────


class TestRatios:
    def test_compute_ratios(self) -> None:
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        closes = {
            "XLY": pd.Series(np.linspace(100, 120, n), index=dates),
            "XLP": pd.Series(np.linspace(100, 110, n), index=dates),
            "IWM": pd.Series(np.linspace(100, 130, n), index=dates),
            "SPY": pd.Series(np.linspace(100, 115, n), index=dates),
            "XLU": pd.Series(np.linspace(100, 105, n), index=dates),
            "XLF": pd.Series(np.linspace(100, 125, n), index=dates),
            "CPER": pd.Series(np.linspace(100, 140, n), index=dates),
            "GLD": pd.Series(np.linspace(100, 115, n), index=dates),
            "USO": pd.Series(np.linspace(100, 110, n), index=dates),
        }
        ratios = _compute_ratios(closes)
        assert "xly_xlp" in ratios
        assert ratios["xly_xlp"] > 0
        assert ratios["cper_gld"] > 0

    def test_ratio_trends(self) -> None:
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        closes = {
            "XLY": pd.Series(np.linspace(100, 150, n), index=dates),
            "XLP": pd.Series(np.linspace(100, 100, n), index=dates),
            "IWM": pd.Series(np.linspace(100, 140, n), index=dates),
            "SPY": pd.Series(np.linspace(100, 130, n), index=dates),
            "XLU": pd.Series(np.linspace(100, 100, n), index=dates),
            "XLF": pd.Series(np.linspace(100, 120, n), index=dates),
            "CPER": pd.Series(np.linspace(100, 160, n), index=dates),
            "GLD": pd.Series(np.linspace(100, 100, n), index=dates),
            "USO": pd.Series(np.linspace(100, 110, n), index=dates),
        }
        trends = _ratio_trends(closes)
        # XLY/XLP: XLY rose (50%), XLP flat (0%) -> ratio should rise
        assert trends["xly_xlp_trend"] > 0
        # CPER/GLD: CPER rose faster than GLD
        assert trends["cper_gld_trend"] > 0

    def test_missing_data(self) -> None:
        ratios = _compute_ratios({})
        assert all(v == 0.0 for v in ratios.values())

        trends = _ratio_trends({})
        assert all(v == 0.0 for v in trends.values())


# ── Test layer 5: Breadth ──────────────────────────────────────


class TestBreadth:
    def test_all_above_dma(self, simple_closes: dict[str, pd.Series]) -> None:
        """If all assets are in uptrend, most should be above DMAs."""
        breadth = _calc_breadth(simple_closes)
        assert breadth["above_50dma"] > 50.0
        assert breadth["above_200dma"] > 50.0

    def test_empty_closes(self) -> None:
        breadth = _calc_breadth({})
        assert breadth["above_50dma"] == 0.0
        assert breadth["above_200dma"] == 0.0

    def test_all_below(self) -> None:
        """If all assets crash, breadth should be 0."""
        n = 250
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        closes = {
            "SPY": pd.Series(np.linspace(100, 50, n), index=dates),
            "XLF": pd.Series(np.linspace(100, 40, n), index=dates),
        }
        breadth = _calc_breadth(closes)
        assert breadth["above_50dma"] == 0.0
        assert breadth["above_200dma"] == 0.0


# ── Test layer 6: Credit scores ────────────────────────────────


class TestCreditScores:
    def test_basic_credit(self) -> None:
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        closes = {
            "HYG": pd.Series(np.linspace(100, 110, n), index=dates),
            "LQD": pd.Series(np.linspace(100, 105, n), index=dates),
            "TLT": pd.Series(np.linspace(100, 95, n), index=dates),
        }
        credit = _calc_credit_scores(closes)
        assert "hyg_tlt_ratio" in credit
        assert "lqd_tlt_ratio" in credit
        # HYG/TLT rising since HYG up, TLT down
        assert "hyg_tlt_trend" in credit
        assert credit["hyg_tlt_trend"] > 0

    def test_missing_ticker(self) -> None:
        closes = {"HYG": pd.Series([100.0])}
        credit = _calc_credit_scores(closes)
        assert credit == {}


# ── Test layer 7: Rates scores ─────────────────────────────────


class TestRatesScores:
    def test_basic_rates(self) -> None:
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        closes = {
            "SHY": pd.Series(np.linspace(100, 101, n), index=dates),
            "IEF": pd.Series(
                np.linspace(100, 95, n), index=dates
            ),  # falling = rising yields
        }
        rates = _calc_rates_scores(closes)
        assert "spread_2s10s" in rates
        assert "yield_trend_10y" in rates
        # IEF fell -> yields rose -> yield_trend should be positive (inverted)
        assert rates["yield_trend_10y"] > 0

    def test_missing_ticker(self) -> None:
        closes = {"SHY": pd.Series([100.0])}
        rates = _calc_rates_scores(closes)
        assert rates == {}


# ── Test layer 8: Regime scoring ───────────────────────────────


class TestScoreRisk:
    def test_risk_on(self) -> None:
        """HYG/TLT rising, VIX low -> risk-on."""
        credit = {"hyg_tlt_trend": 6.0}
        score = _score_risk(credit, volatility_px=12.0)
        assert score > 60.0

    def test_risk_off(self) -> None:
        """HYG/TLT falling, VIX high -> risk-off."""
        credit = {"hyg_tlt_trend": -6.0}
        score = _score_risk(credit, volatility_px=35.0)
        assert score < 40.0

    def test_neutral(self) -> None:
        credit: dict[str, float] = {}
        score = _score_risk(credit, volatility_px=None)
        assert score == 50.0


class TestScoreGrowth:
    def test_pro_cyclical(self) -> None:
        """XLY/XLP rising, IWM/SPY rising, cyclical leaders."""
        trends = {"xly_xlp_trend": 6.0, "iwm_spy_trend": 4.0, "xli_xlu_trend": 4.0}
        leadership = [("XLF", 0.1), ("XLK", 0.08), ("XLI", 0.06)]
        score = _score_growth(trends, leadership)
        assert score > 60.0

    def test_defensive(self) -> None:
        """XLY/XLP falling, cyclical leaders absent."""
        trends = {"xly_xlp_trend": -6.0, "iwm_spy_trend": -4.0, "xli_xlu_trend": -4.0}
        leadership = [("XLV", 0.1), ("XLP", 0.08), ("XLU", 0.06)]
        score = _score_growth(trends, leadership)
        assert score < 45.0


class TestScoreInflation:
    def test_rising_inflation(self) -> None:
        trends = {"cper_gld_trend": 6.0, "uso_gld_trend": 6.0}
        closes = {"XLE": pd.Series(np.linspace(100, 130, 100))}
        score = _score_inflation(trends, closes)
        assert score > 55.0

    def test_falling_inflation(self) -> None:
        trends = {"cper_gld_trend": -6.0, "uso_gld_trend": -6.0}
        score = _score_inflation(trends, {})
        assert score < 45.0


class TestScoreBreadth:
    def test_broad_participation(self) -> None:
        breadth = {"above_50dma": 85.0, "above_200dma": 80.0}
        score = _score_breadth(breadth)
        assert score > 75.0

    def test_poor_breadth(self) -> None:
        breadth = {"above_50dma": 20.0, "above_200dma": 15.0}
        score = _score_breadth(breadth)
        assert score < 30.0


class TestScoreLiquidity:
    def test_accommodative(self) -> None:
        """Yields falling, credit spreads narrowing."""
        rates = {"yield_trend_10y": 6.0}  # positive = yields falling
        credit = {"hyg_tlt_trend": 6.0}  # positive = spreads narrowing
        score = _score_liquidity(rates, credit)
        assert score > 60.0

    def test_tight(self) -> None:
        """Yields rising, credit spreads widening."""
        rates = {"yield_trend_10y": -6.0}  # negative = yields rising
        credit = {"hyg_tlt_trend": -6.0}  # negative = spreads widening
        score = _score_liquidity(rates, credit)
        assert score < 40.0


# ── Test layer 9: Regime classification ────────────────────────


class TestClassifyRegime:
    def test_expansion(self) -> None:
        regime, conf = _classify_regime(
            risk_score=80.0,
            growth_score=85.0,
            inflation_score=50.0,
            breadth_score=80.0,
            liquidity_score=70.0,
        )
        assert regime == "Expansion"
        assert conf > 50.0

    def test_crisis(self) -> None:
        regime, conf = _classify_regime(
            risk_score=20.0,
            growth_score=15.0,
            inflation_score=40.0,
            breadth_score=20.0,
            liquidity_score=25.0,
        )
        assert regime == "Crisis"
        assert conf > 30.0

    def test_inflationary_boom(self) -> None:
        regime, _ = _classify_regime(
            risk_score=60.0,
            growth_score=90.0,
            inflation_score=90.0,
            breadth_score=70.0,
            liquidity_score=70.0,
        )
        assert regime == "Inflationary_Boom"

    def test_late_cycle(self) -> None:
        regime, _ = _classify_regime(
            risk_score=40.0,
            growth_score=35.0,
            inflation_score=70.0,
            breadth_score=50.0,
            liquidity_score=30.0,
        )
        assert regime == "Late_Cycle"


# ── Test layer 10: Cross-asset confirmation ────────────────────


class TestCrossAssetConfirmation:
    def test_strong_confirmation(self) -> None:
        trends = {
            "iwm_spy_trend": 5.0,
            "xly_xlp_trend": 4.0,
            "cper_gld_trend": 3.0,
            "hyg_tlt_trend": 5.0,
        }
        conf = _cross_asset_confirmation(trends, risk_score=70.0, growth_score=70.0)
        assert conf > 50.0

    def test_no_confirmation(self) -> None:
        trends = {
            "iwm_spy_trend": 5.0,
            "xly_xlp_trend": -4.0,
            "cper_gld_trend": -3.0,
            "hyg_tlt_trend": -5.0,
        }
        conf = _cross_asset_confirmation(trends, risk_score=70.0, growth_score=30.0)
        assert conf < 50.0


# ── Test CycleScreen class ─────────────────────────────────────


class TestCycleScreen:
    def test_make_returns_screen(self) -> None:
        from screens.cycle_screener import make

        screen = make(symbols=["SPY"], params={})
        assert isinstance(screen, CycleScreen)

    def test_compute_returns_screen_result_contract(self) -> None:
        screen = CycleScreen(symbols=["SPY"], params={})
        # compute() will try to fetch from DB — this tests the contract
        # that it always returns a valid ScreenResult even on failure
        candles = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
            },
            index=pd.DatetimeIndex(["2023-01-01"]),
        )
        result = screen.compute("SPY", candles)
        assert isinstance(result, ScreenResult)
        assert result.signal in ("long", "short", "neutral")

    def test_compute_second_call_returns_noop(self) -> None:
        screen = CycleScreen(symbols=["SPY"], params={})
        screen._computed = True  # Simulate first call
        candles = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
            },
            index=pd.DatetimeIndex(["2023-01-01"]),
        )
        result = screen.compute("SPY", candles)
        assert result.signal == "neutral"
        assert result.score == 0.0
        assert result.metadata.get("reason") == "macro_result_already_produced"

    def test_rank_sorts_by_score(self) -> None:
        screen = CycleScreen(symbols=[], params={})
        results = [
            ScreenResult("A", "neutral", 0.5, 100.0, {}),
            ScreenResult("B", "long", 0.8, 100.0, {}),
            ScreenResult("C", "neutral", 0.2, 100.0, {}),
        ]
        ranked = screen.rank(results)
        assert ranked[0].score >= ranked[1].score >= ranked[2].score

    def test_rank_sinks_noop(self) -> None:
        screen = CycleScreen(symbols=[], params={})
        results = [
            ScreenResult("A", "long", 0.8, 100.0, {}),
            ScreenResult("B", "neutral", 0.0, 0.0, {"reason": "no_universe_data"}),
            ScreenResult("C", "long", 0.6, 100.0, {}),
        ]
        ranked = screen.rank(results)
        assert ranked[0].symbol == "A"
        assert ranked[-1].symbol == "B"  # noop sinks to bottom


# ── Helper to test ScreenResult creation ───────────────────────

from src.screen.types import ScreenResult


def test_screen_result_contract() -> None:
    """Verify ScreenResult is a frozen dataclass with correct fields."""
    r = ScreenResult(
        symbol="MACRO",
        signal="long",
        score=0.75,
        price=100.0,
        metadata={"regime": "Expansion", "confidence": 80.0},
    )
    assert r.symbol == "MACRO"
    assert r.signal == "long"
    assert r.score == 0.75
    assert r.price == 100.0
    assert r.metadata["regime"] == "Expansion"


# ── Test DEFAULTS export ───────────────────────────────────────


def test_defaults_exist() -> None:
    from screens.cycle_screener import DEFAULTS

    assert isinstance(DEFAULTS, dict)
    assert "lookback_days" in DEFAULTS
    assert "rs_window" in DEFAULTS


# ── Test make() factory ────────────────────────────────────────


def test_make_returns_screen_fn() -> None:
    from screens.cycle_screener import make

    screen = make(symbols=["SPY"], params={"lookback_days": 504})
    assert hasattr(screen, "compute")
    assert hasattr(screen, "rank")
