"""Tests for the output-boundary renderers (JSON/JSONL from BacktestResults)."""

import pandas as pd

from src.bt.output import (
    trade_json,
    equity_points,
    render_result_json,
    render_result_jsonl,
)
from src.bt.state import (
    ActionType,
    PortfolioResult,
    Trade,
    TradeStatus,
    TradeExitReason,
)


def _ts(s: str) -> pd.Timestamp:
    """Parse a datestring into a non-NaT Timestamp for typing clarity."""
    t = pd.Timestamp(s)
    assert isinstance(t, pd.Timestamp) and not pd.isna(t)
    return t


def _trade(status=TradeStatus.closed) -> Trade:
    return Trade(
        entry_time=_ts("2021-01-05 09:30:00"),
        entry_price=100.5,
        exit_time=_ts("2021-01-06 10:00:00"),
        exit_price=105.0,
        last_price=105.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=95.0,
        take_profit=110.0,
        pnl=45.0,
        commission=0.5,
        slippage=0.1,
        reason="breakout",
        status=status,
        close_reason=TradeExitReason.tp,
        position_id="AAPL_123",
    )


def _pf():
    equity = pd.Series(
        [100000.0, 101000.0, 100500.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    )
    return PortfolioResult(
        total_return=0.005,
        sharpe_ratio=1.2,
        trades=(_trade(),),
        equity_curve=equity,
        annual_return=0.05,
    )


def test_trade_json_flattens_enums_and_value():
    j = trade_json(_trade())
    assert j["position"] == "long"  # ActionType enum -> plain value
    assert j["status"] == "closed"  # TradeStatus enum -> plain value
    assert j["close_reason"] == "tp"  # TradeExitReason enum -> plain value
    assert j["symbol"] == "AAPL"
    assert isinstance(j["pnl"], float)
    assert j["position_id"] == "AAPL_123"


def test_trade_json_unclosed_exit_is_none():
    t = _trade()
    t.exit_time = None
    t.exit_price = None
    j = trade_json(t)
    assert j["exit_time"] is None
    assert j["exit_price"] is None


def test_equity_points_dates_and_floats():
    pts = equity_points(_pf().equity_curve)
    assert len(pts) == 3
    assert pts[0]["ts"] == "2021-01-01 00:00:00"
    assert abs(pts[0]["equity"] - 100000.0) < 1e-9


def test_render_result_json_shape():
    r = render_result_json(type("R", (), {"pf": _pf(), "benchmark_curves": {}})())
    assert set(r) == {"metrics", "trades", "equity_curve", "benchmark_curves"}
    assert r["metrics"]["total_return"] == 0.005
    assert len(r["trades"]) == 1
    # benchmark_curves keyed by symbol -> list of points
    assert r["benchmark_curves"] == {}


def test_render_result_jsonl_ends_with_summary_record():
    obj = type("R", (), {"pf": _pf(), "benchmark_curves": {}})()
    lines = render_result_jsonl(obj)
    # 3 equity points + 1 summary
    assert len(lines) == 4
    assert set(lines[0]) == {"ts", "equity"}
    assert set(lines[-1]) == {"metrics", "trades"}
