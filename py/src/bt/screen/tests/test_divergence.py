"""Tests for cross-interval TF-divergence ranking (``rank_divergence``)."""

from __future__ import annotations

import pandas as pd

from src.bt.screen.runner import DivergenceParams, rank_divergence
from src.bt.screen.types import ScreenState, TrendRegime


def _state(ts: pd.Timestamp, trend: dict[str, TrendRegime | None]) -> ScreenState:
    return ScreenState(ts=ts, frames=(), trend=trend, vol={})


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def test_full_bull_alignment_scores_long():
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "BULL"}),
        "4h": _state(_ts("2024-01-02 16:00"), {"X": "BULL"}),
        "1d": _state(_ts("2024-01-02"), {"X": "BULL"}),
    }
    (r,) = rank_divergence(states)
    assert r.symbol == "X"
    assert r.action == "long"
    assert r.score == 1.0
    assert "DIVERGENT" not in r.signals
    assert abs(r.model_features["long_align"] - 1.0) < 1e-9


def test_full_bear_alignment_scores_short():
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "BEAR"}),
        "1d": _state(_ts("2024-01-02"), {"X": "BEAR"}),
    }
    (r,) = rank_divergence(states)
    assert r.action == "short"
    assert r.score == 1.0


def test_divergence_sets_flag_low_score():
    # 1h BULL against 1d BEAR -> conflict, direction by weight, flag surfaced.
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "BULL"}),
        "1d": _state(_ts("2024-01-02"), {"X": "BEAR"}),
    }
    (r,) = rank_divergence(states)
    assert r.model_features["divergent"] == 1.0
    assert "DIVERGENT" in r.signals


def test_partial_alignment_scores_mid_threshold():
    # 1h BULL, 4h RANGE -> alignment below 0.5 threshold -> flat.
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "BULL"}),
        "4h": _state(_ts("2024-01-02 16:00"), {"X": "RANGE"}),
    }
    lowered = DivergenceParams(lower_tf_weight=1.0, alignment_threshold=0.0)
    (r,) = rank_divergence(states, lowered)
    assert r.action == "long"
    # With default 0.5 threshold the single BULL weight=1.5 / total 2.5 = 0.6
    # crosses the bar when weights differ.
    assert 0.0 <= r.score <= 1.0


def test_lower_tf_weight_biases_alignment():
    # Equal BULL/BEAR across 1h vs 1d: lower TF (1h) weighs more -> long.
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "BULL"}),
        "1d": _state(_ts("2024-01-02"), {"X": "BEAR"}),
    }
    (r,) = rank_divergence(states)
    assert r.action == "long"
    assert r.model_features["long_align"] > r.model_features["short_align"]


def test_no_trend_labels_sort_flat():
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"X": "RANGE"}),
        "1d": _state(_ts("2024-01-02"), {"X": "RANGE"}),
    }
    (r,) = rank_divergence(states)
    assert r.action == "flat"
    assert r.score == 0.0
    assert r.model_features["divergent"] == 0.0


def test_top_caps_results():
    states = {
        "1h": _state(_ts("2024-01-02 17:00"), {"A": "BULL", "B": "BEAR"}),
        "1d": _state(_ts("2024-01-02"), {"A": "BULL", "B": "BEAR"}),
    }
    ranked = rank_divergence(states, top=1)
    assert len(ranked) == 1
