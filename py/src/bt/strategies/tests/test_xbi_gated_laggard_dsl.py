"""Pure-helper tests for the XBI-gated laggard basket (``xbi_gated_laggard_dsl``)."""

from __future__ import annotations

import numpy as np

from src.bt.strategies.xbi_gated_laggard_dsl import (
    _select_worst_laggards,
    _tail_log_ret,
)


def test_tail_log_ret_empty_and_short():
    """NaN for empty or too-short close arrays (windows cannot form)."""
    assert np.isnan(_tail_log_ret(np.array([], dtype=float), 63))
    assert np.isnan(_tail_log_ret(np.array([10.0, 11.0]), 5))


def test_tail_log_ret_value():
    """Exact log-return over the trailing window."""
    closes = np.array([1.0] * 10 + [np.e], dtype=float)
    assert np.isclose(_tail_log_ret(closes, 9), 1.0)


def test_tail_log_ret_nonpositive_nan():
    """A non-positive close inside the window never yields a finite return."""
    # window start lands on 0.0 -> log(4/0) must be NaN, not +/-inf
    closes = np.array([1.0, 0.0, 2.0, 4.0], dtype=float)
    assert np.isnan(_tail_log_ret(closes, 2))
    # final close is 0.0 -> NaN
    closes2 = np.array([1.0, 2.0, 4.0, 0.0], dtype=float)
    assert np.isnan(_tail_log_ret(closes2, 2))


def test_select_worst_keeps_lowest_excluding_gate():
    """Worst-laggard selection sorts ascending and never picks the gate."""
    retmap = {"AA": -0.05, "BB": -0.22, "CC": 0.01, "gxbi": -1.0, "DD": -0.40}
    picks = _select_worst_laggards(retmap, {"gxbi"}, tail_n=2)
    assert picks == ["DD", "BB"], "worst two real names, gate excluded"
    assert "gxbi" not in picks


def test_select_worst_respects_tail_cap():
    """Never returns more than ``tail_n`` nor more than panel members."""
    retmap = {f"S{i}": float(-i) for i in range(20)}  # distinct descending winners
    picks = _select_worst_laggards(retmap, set(), tail_n=5)
    assert len(picks) == 5
    assert picks[0] == "S19"  # most negative (worst) laggard first


def test_select_worst_empty():
    """Empty ranking or tail_n<=0 yields no picks (never a degenerate basket)."""
    assert _select_worst_laggards({}, {"XBI"}, tail_n=5) == []
    assert _select_worst_laggards({"AA": -0.1}, set(), tail_n=-1) == []
