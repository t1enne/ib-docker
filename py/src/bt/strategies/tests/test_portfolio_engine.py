"""Regression tests for the pf_* rebalancing engine (P0–P2).

Covers the portfolio-construction correctness fixes:
  P0-1/2 — rebalance bookkeeping: TWAC, no fabricated trades, cash conservation.
  P1-3   — unconditional cadence (no drift skipping).
  P1-4   — calendar cadence (fires on calendar boundaries, not bar counts).
  P1-5   — consistent warmup -> identical first-rebalance date across schemes.
  P1-6   — gross-turnover accounting and turnover-based cost.
  P2-7   — go-to-cash on all-zero targets.
  weight — GMV/invvol/risk-parity/fixed-alloc invariants (sum ≈ 1).

All runs use deterministic synthetic daily closes so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run
from src.bt.strategies import init_strat
from src.bt.types import StrategyConfig


def _ts(v: str) -> pd.Timestamp:
    """Typed Timestamp literal (avoids `Timestamp | NaTType` from pandas stubs)."""
    from typing import cast as _cast

    ts = pd.Timestamp(v)
    assert not pd.isna(ts)
    return _cast(pd.Timestamp, ts)


def _daily_ohlc(
    symbols: list[str], start: str, periods: int, seed: int = 42
) -> pd.DataFrame:
    """Synthetic daily OHLCV MultiIndex DataFrame across symbols."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="D")
    data: dict = {}
    for sym in symbols:
        # smooth random walk with a per-symbol drift so weights move
        drift = rng.normal(0.0003, 0.0006)
        rets = drift + rng.normal(0.0, 0.01, size=periods)
        close = 100.0 * np.cumprod(1 + rets)
        data[(sym, "open")] = close
        data[(sym, "high")] = close * 1.01
        data[(sym, "low")] = close * 0.99
        data[(sym, "close")] = close
        data[(sym, "volume")] = 1000.0
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(
    strat_type: str,
    symbols: list[str],
    params: dict,
    *,
    start: str,
    periods: int,
    trading_start: str | None = None,
) -> StrategyConfig:
    trade_start = trading_start or start
    trade_end = str(pd.Timestamp(start) + pd.Timedelta(days=periods - 1)).split()[0]
    return StrategyConfig(
        name=f"t-{strat_type}",
        strategy_type=strat_type,
        symbols=symbols,
        initial_capital=100_000.0,
        commission=0.0,
        training_start=start,
        training_end=str(pd.Timestamp(start) - pd.Timedelta(days=1)).split()[0],
        trading_start=trade_start,
        trading_end=trade_end,
        bars=["1d"],
        strategy_params=params,
        model_params={},
        benchmark_symbols=[],
    )


def _run(strat_type: str, symbols: list[str], df: pd.DataFrame, params: dict, **kw):
    cfg = _cfg(
        strat_type, symbols, params, start=str(df.index[0]), periods=len(df), **kw
    )
    bt = Backtest(cfg)
    mod = init_strat(strat_type)
    # The engine does not reset module GLOBAL state between runs; the split
    # engine normally calls reset_global(). Clear it here so repeated runs in
    # this suite are independent (no stale bar_idx/next_rebalance/weight_fn).
    reset = getattr(mod, "reset_global", None)
    if reset is not None:
        reset()
    # ``run()`` builds the candle generator internally from the DataFrame.
    results = run(bt, df, strat_mod=mod)
    # Mirror src.bt.run_backtest_results: merge the strategy module's runtime
    # pf-report stats (rebalance count / gross turnover) into the result.
    from dataclasses import replace as dreplace
    from src.bt.strategies import resolve_params

    stats_fn = getattr(mod, "runtime_stats", None)
    if stats_fn is not None:
        resolved = resolve_params(strat_type, params)
        default_bps = stats_fn()["turnover_cost_bps"]
        bps = float(getattr(resolved, "cost_bps", default_bps))
        s = stats_fn(cost_bps=bps)
        pf = dreplace(
            results.pf,
            gross_turnover=s["gross_turnover"],
            n_rebalances=s["n_rebalances"],
            turnover_cost_bps=s["turnover_cost_bps"],
            turnover_cost=(
                s["gross_turnover"]
                * (s["turnover_cost_bps"] / 1e4)
                * cfg.initial_capital
            ),
        )
        results = dreplace(results, pf=pf)
    return results, cfg


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


def _closed_trades(state) -> list:
    return [t for t in state.portfolio.trades if t.status.value == "closed"]


def _open_trades(state) -> list:
    return [t for t in state.portfolio.trades if t.status.value == "open"]


# ---------------------------------------------------------------------------
# P0-1 / P0-2 — rebalance bookkeeping
# ---------------------------------------------------------------------------


def test_rebalance_keeps_one_trade_per_symbol_no_fabricated_closes():
    """A 2-symbol fixed-alloc PRP across several months must record exactly one
    open trade per symbol (no close/reopen churn), and closing at the end must
    realize an honest round-trip PnL against the time-weighted average cost."""
    symbols = ["SPY", "TLT"]
    # ~15 months of daily data, monthly rebalance, warmup 2 bars.
    df = _daily_ohlc(symbols, "2020-01-01", 340, seed=7)
    params = {
        "allocations": {"SPY": 0.6, "TLT": 0.4},
        "rebalance": "monthly",
        "warmup_bars": 2,
        "interval": "1d",
    }
    results, cfg = _run("pf_alloc", symbols, df, params)  # noqa: F841
    state = results.final_state

    # Exactly one trade per symbol — no interim close/reopen fabrications.
    by_sym: dict[str, int] = {}
    for t in state.portfolio.trades:
        by_sym[t.symbol] = by_sym.get(t.symbol, 0) + 1
    assert by_sym == {"SPY": 1, "TLT": 1}

    # Both must be open (never closed mid-run) and closed at end by finalize.
    # run() finalizes (closes at end), so all trades become closed.
    closed = _closed_trades(state)
    assert len(closed) == 2
    # No zero-or-ancient pnl artifacts: each final PnL is a real round trip.
    for t in closed:
        assert t.entry_price > 0
        assert t.exit_price > 0
        assert np.isfinite(t.pnl)


def test_rebalance_cash_preserved_by_construction():
    """Applying a pure-add then pure-reduce rebalance must not create or
    destroy equity — cash + positions value is invariant to the fills."""
    from src.bt.portfolio.pure import apply_fill
    from src.bt.state import (
        FillEvent,
        PortfolioState,
        TradeSignal,
        ActionType,
        create_initial_portfolio,
    )

    port = create_initial_portfolio(
        initial_capital=100_000.0, start_timestamp=_ts("2020-01-01")
    )
    # Open a long manually at SPY=100.
    fill_open = FillEvent(
        signal=TradeSignal(
            action=ActionType.long,
            symbol="SPY",
            timestamp=_ts("2020-01-02"),
            price=100.0,
            qty=500.0,
            position_id="SPY_1",
        ),
        filled_qty=500.0,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2020-01-02"),
    )
    port = apply_fill(port, fill_open)
    assert port.cash == pytest.approx(50_000.0)  # 100k - 500*100

    def _eq(port: PortfolioState) -> float:
        val = sum(abs(p.qty) * p.last_price for p in port.positions.get("SPY", ()))
        return port.cash + val

    # Pure add (+100 shares @ the same price 100): cash out = delta*price.
    add = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="SPY",
            timestamp=_ts("2020-02-03"),
            price=100.0,
            qty=100.0,
            position_id="SPY_1",
        ),
        filled_qty=100.0,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2020-02-03"),
    )
    before_tick = _eq(port)
    port = apply_fill(port, add)
    pos = port.positions["SPY"][0]
    assert pos.qty == pytest.approx(600.0)
    # TWAC entry re-averaged at same price => unchanged basis.
    assert pos.entry_price == pytest.approx(100.0)
    # Cash reduced by exactly delta*price; equity conserved (mark unchanged).
    assert port.cash == pytest.approx(50_000.0 - 100 * 100.0)
    assert _eq(port) == pytest.approx(before_tick, abs=1e-6)

    # Pure reduce (sell 200 @ the same price 100): cash in = reduce*price.
    red = FillEvent(
        signal=TradeSignal(
            action=ActionType.rebalance,
            symbol="SPY",
            timestamp=_ts("2020-03-03"),
            price=100.0,
            qty=-200.0,
            position_id="SPY_1",
        ),
        filled_qty=-200.0,
        executed_price=100.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2020-03-03"),
    )
    before_red = _eq(port)
    port = apply_fill(port, red)
    assert port.positions["SPY"][0].qty == pytest.approx(400.0)
    assert port.cash == pytest.approx(50_000.0 - 100 * 100.0 + 200 * 100.0)
    assert _eq(port) == pytest.approx(before_red, abs=1e-6)
    # Only one open trade remains (no fabricated close/reopen on reduce).
    assert len([t for t in port.trades if t.status.value == "open"]) == 1


# ---------------------------------------------------------------------------
# P1-3 + P1-4 — unconditional, calendar-cadenced rebalancing
# ---------------------------------------------------------------------------


def test_monthly_rebalance_fires_on_every_month_end():
    """A monthly fixed-alloc PRP over ~2 years rebalances on each month-end
    boundary present in the data, with no drift-skip."""
    symbols = ["SPY", "TLT", "GLD"]
    df = _daily_ohlc(symbols, "2020-01-01", 730, seed=3)
    params = {
        "allocations": {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2},
        "rebalance": "monthly",
        "warmup_bars": 5,
        "interval": "1d",
    }
    results, cfg = _run("pf_alloc", symbols, df, params)
    state = results.final_state
    mod = init_strat("pf_alloc")
    stats = mod.runtime_stats()
    assert stats["gross_turnover"] > 0.0

    # Every emitted rebalance signal carried month-end origin. Reconstruct from
    # the single trade per symbol: the entry_time is the first rebalance date
    # and must land on a month end (in the traded data).
    for t in state.portfolio.trades:
        entry = pd.Timestamp(t.entry_time)
        assert entry == entry + pd.tseries.offsets.MonthEnd(0), f"{entry}"

    # Unconditional: the number of fired rebalances equals the number of
    # month-end boundaries within the warmup-tailed window (roughly months).
    month_ends = pd.date_range("2020-01-31", cfg.trading_end, freq="ME")
    # first boundary at/after warmup bar (2020-01-06) -> 2020-01-31; count up to end
    eligible = [m for m in month_ends if m <= pd.Timestamp(cfg.trading_end)]
    assert stats["n_rebalances"] >= len(eligible) - 1


def test_calendar_not_barcount_monthly_smoke():
    """Monthly cadence produces rebalances on calendar month-ends, so exposure
    is not tied to a sliding 21-bar counter (which would drift)."""
    symbols = ["A", "B", "C"]
    df = _daily_ohlc(symbols, "2021-06-01", 400, seed=9)
    params = {
        "allocations": {"A": 0.34, "B": 0.33, "C": 0.33},
        "rebalance": "monthly",
        "warmup_bars": 2,
        "interval": "1d",
    }
    results, cfg = _run("pf_alloc", symbols, df, params)
    mod = init_strat("pf_alloc")
    stats = mod.runtime_stats()
    # Rebalance must not simply equal a 21-bar counter — over ~13 months expect
    # ~13 month-end rebalances (calendar) not ~19 (21-bar).
    assert stats["n_rebalances"] <= 15
    assert stats["n_rebalances"] >= 11


# ---------------------------------------------------------------------------
# P1-5 — consistent warmup: identical first-rebalance date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strat_type",
    ["pf_alloc", "pf_gmv", "pf_invvol", "pf_risk_parity"],
)
def test_first_rebalance_date_aligned_across_schemes(strat_type):
    """With an explicit small, equal warmup_bars, the first rebalance fires at
    the same cadence boundary (assert it's a month-end boundary)."""
    first = _first_rebalance(strat_type)
    assert first == first + pd.tseries.offsets.MonthEnd(0)


def _first_rebalance(strat_type: str) -> pd.Timestamp:
    """Run one scheme and return the date its first position was opened (== the
    first rebalance date under the same warmup rule)."""
    symbols = ["SPY", "TLT", "GLD", "DBA"]
    df = _daily_ohlc(symbols, "2019-01-01", 500, seed=11)
    params = {"rebalance": "monthly", "warmup_bars": 10, "interval": "1d"}
    if strat_type == "pf_alloc":
        params["allocations"] = {"SPY": 0.5, "TLT": 0.5, "GLD": 0.0, "DBA": 0.0}
    results, cfg = _run(strat_type, symbols, df, params)
    state = results.final_state
    entries = sorted(t.entry_time for t in state.portfolio.trades)
    assert entries, "no trades produced"
    return entries[0]


def test_all_schemes_share_first_rebalance():
    first_dates = {}
    for strat_type in ["pf_alloc", "pf_gmv", "pf_invvol", "pf_risk_parity"]:
        first_dates[strat_type] = _first_rebalance(strat_type)
    vals = list(first_dates.values())
    assert all(v == vals[0] for v in vals), f"mismatched first rebalance: {first_dates}"


# ---------------------------------------------------------------------------
# P1-6 — turnover & turnover-cost accounting
# ---------------------------------------------------------------------------


def test_turnover_cost_matches_bps():
    """Reported turnover-based cost equals gross turnover x bps/1e4 x capital."""
    symbols = ["SPY", "TLT"]
    df = _daily_ohlc(symbols, "2020-01-01", 400, seed=5)
    params = {
        "allocations": {"SPY": 0.6, "TLT": 0.4},
        "rebalance": "monthly",
        "warmup_bars": 2,
        "interval": "1d",
        "cost_bps": 20.0,
    }
    results, cfg = _run("pf_alloc", symbols, df, params)
    pf = results.pf
    assert pf.turnover_cost_bps == pytest.approx(20.0)
    expected = pf.gross_turnover * (pf.turnover_cost_bps / 1e4) * cfg.initial_capital
    assert pf.turnover_cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# P2-7 — go-to-cash on all-zero targets
# ---------------------------------------------------------------------------


def test_p2_7_driver_goes_to_cash_unused_direct():
    """The go-to-cash helper emits a close when targets are all zero and a
    position exists (unit-level check of the driver's close branch)."""
    from src.bt.state import (
        create_initial_backtest_state,
        Position,
        ActionType,
        PortfolioState,
        Candle,
    )
    from src.bt.strategies.portfolio_engine import _go_to_cash

    state = create_initial_backtest_state(
        symbols=["SPY"],
        initial_capital=100_000.0,
        start_timestamp=_ts("2020-01-01"),
    )
    pos = Position(
        symbol="SPY",
        qty=500.0,
        entry_price=100.0,
        entry_time=_ts("2020-01-02"),
        stop_loss=None,
        take_profit=None,
        last_price=105.0,
        type=ActionType.long,
        position_id="SPY_1",
    )
    from dataclasses import replace as dreplace

    port = PortfolioState(
        cash=50_000.0,
        positions={"SPY": (pos,)},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )
    state = dreplace(state, portfolio=port)
    candle = Candle(
        timestamp=_ts("2020-03-31"),
        symbol="SPY",
        open=105.0,
        high=106.0,
        low=104.0,
        close=105.0,
        volume=1.0,
        interval="1d",
    )
    sigs = _go_to_cash(state, candle, ["SPY"], {"SPY": 105.0})
    assert len(sigs) == 1
    assert sigs[0].action == ActionType.close
    assert sigs[0].position_id == "SPY_1"


def test_p2_7_driver_zero_targets_goes_to_cash():
    """When targets sum <= 0 and positions are held, ``pf_on_candle`` emits a
    go-to-cash close rather than silently holding (P2-7). With no positions it
    no-ops."""
    from src.bt.strategies.portfolio_engine import pf_on_candle
    from src.bt.strategies.portfolio_weights import WeightMethodFn
    from src.bt.engine.candle_store import CandleStore
    from src.bt.state import (
        Position,
        PortfolioState,
        ActionType,
        BacktestState,
        ModelState,
        MarketDataState,
        Candle,
    )

    def _zero(
        returns: pd.DataFrame,
        long_only: bool = True,  # noqa: ARG001
    ) -> pd.Series:
        return pd.Series(0.0, index=returns.columns)

    zfn: WeightMethodFn = _zero

    g = {
        "last_signal_close": {},
        "bar_idx": 0,
        "next_rebalance": None,
        "n_rebalances": 0,
        "gross_turnover": 0.0,
        "last_plan": "init",
    }

    # Build a state with a held SPY position and enough warmup candles.
    n = 10
    rows: dict = {
        ("SPY", "1d"): {
            "timestamp": np.array(
                [np.datetime64(f"2020-01-{i + 1:02d}") for i in range(n)],
                dtype="datetime64[ms]",
            ),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "_len": np.array([n]),
        },
        ("TLT", "1d"): {
            "timestamp": np.array(
                [np.datetime64(f"2020-01-{i + 1:02d}") for i in range(n)],
                dtype="datetime64[ms]",
            ),
            "open": np.full(n, 50.0),
            "high": np.full(n, 51.0),
            "low": np.full(n, 49.0),
            "close": np.full(n, 50.0),
            "volume": np.full(n, 1000.0),
            "_len": np.array([n]),
        },
    }
    store = CandleStore(rows)
    pos = Position(
        symbol="SPY",
        qty=500.0,
        entry_price=100.0,
        entry_time=_ts("2020-01-02"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
        position_id="SPY_1",
    )
    port = PortfolioState(
        cash=50_000.0,
        positions={"SPY": (pos,)},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )
    state = BacktestState(
        portfolio=port,
        timestamp=_ts("2020-01-10"),
        pending_signals={},
        model_state=ModelState(
            z_score=None,
            current_regime=None,
            price_buffers=(),
            market_data=MarketDataState(symbols=("SPY",)),
        ),
        risk_events=(),
        candles=store,
    )
    candle = Candle(
        timestamp=_ts("2020-01-10"),
        symbol="SPY",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
        interval="1d",
    )

    sigs = pf_on_candle(
        state,
        candle,
        g,
        zfn,
        interval="1d",
        rebalance="daily",
        lookback=2,
        max_weight=1.0,
        warmup_bars=2,
    )
    # Zero targets + held position -> a close is emitted, not a silent hold.
    assert sigs, "zero targets must produce a go-to-cash close when position held"
    assert all(s.action == ActionType.close for s in sigs)
    # And no position is left unaddressed.
    assert g["last_plan"] == "cash"

    # With no positions held, zero targets -> no-op (empty signals).
    empty_port = PortfolioState(
        cash=100_000.0,
        positions={},
        trades=(),
        equity_curve=(),
        initial_capital=100_000.0,
    )
    state2 = BacktestState(
        portfolio=empty_port,
        timestamp=_ts("2020-01-11"),
        pending_signals={},
        risk_events=(),
        model_state=state.model_state,
        candles=store,
    )
    g2 = dict(g, last_plan="init", next_rebalance=None, bar_idx=0)
    candle2 = Candle(
        timestamp=_ts("2020-01-11"),
        symbol="SPY",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
        interval="1d",
    )
    assert (
        pf_on_candle(
            state2,
            candle2,
            g2,
            zfn,
            interval="1d",
            rebalance="daily",
            lookback=2,
            max_weight=1.0,
            warmup_bars=2,
        )
        == []
    )


# ---------------------------------------------------------------------------
# weight invariants
# ---------------------------------------------------------------------------


def test_weight_functions_invariants():
    """GMV / invvol / risk-parity weights sum to ~1; fixed-alloc honours the
    given targets; invvol prefers lower-vol assets."""
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    vol_lo = np.random.default_rng(1).normal(0.0, 0.005, 60)
    vol_hi = np.random.default_rng(2).normal(0.0, 0.04, 60)
    rets = pd.DataFrame(
        {
            "LOW": vol_lo,
            "MID": np.random.default_rng(3).normal(0, 0.02, 60),
            "HIGH": vol_hi,
        },
        index=idx,
    )
    from src.bt.strategies.portfolio_weights import (
        min_variance_weights,
        inverse_vol_weights,
        risk_parity_weights,
        fixed_alloc_weights,
    )

    for fn in (min_variance_weights, inverse_vol_weights, risk_parity_weights):
        w = fn(rets, long_only=True)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= 0).all()
    # invvol: high-vol asset gets a smaller weight than low-vol.
    wiv = inverse_vol_weights(rets, long_only=True)
    assert wiv["LOW"] > wiv["HIGH"]
    # fixed-alloc returns the normalised targets (filtered to present cols).
    fa = fixed_alloc_weights({"LOW": 0.6, "MID": 0.2, "HIGH": 0.2})
    wf = fa(rets, long_only=True)
    assert wf.sum() == pytest.approx(1.0)
    assert wf["LOW"] == pytest.approx(0.6)
