"""Tests for TaContext — prefetched cursor-safe indicator context (DSL Option C)."""

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.candle_store import CandleStore
from src.bt.strategies.ta_context import TaContext, init_ta


def _df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    data = {}
    for sym in ("AAPL", "MSFT"):
        data[(sym, "open")] = close - 0.5
        data[(sym, "high")] = close + 1.0
        data[(sym, "low")] = close - 1.0
        data[(sym, "close")] = close
        data[(sym, "volume")] = np.full(n, 1000.0)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _store(df: pd.DataFrame, symbols: tuple[str, ...]) -> CandleStore:
    """Build a CandleStore whose rows exactly mirror ``df`` (base interval '1d')."""
    n = len(df)
    rows: dict = {}
    for sym in symbols:
        arr = {
            k: np.empty(n) if k != "_len" else np.array([n])
            for k in (
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "_len",
            )
        }
        arr["timestamp"] = df.index.to_numpy().astype("datetime64[ms]")
        for f in ("open", "high", "low", "close", "volume"):
            arr[f] = df[(sym, f)].to_numpy(dtype=float)
        rows[(sym, "1d")] = arr
    return CandleStore(rows)


def _ta(df, symbols) -> tuple[TaContext, CandleStore]:
    ta = TaContext.from_data(df, symbols, "1d")
    store = _store(df, symbols)
    ta.bind(store)
    return ta, store


def test_indicator_computed_once_per_series():
    df = _df()
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[50])
    f1 = ta.ema("AAPL", 9)
    assert f1[-1] == f1[-1]  # not NaN at a warm cursor
    assert ta.compute_count == 1
    # Re-reading the same key is a cache hit -> count unchanged.
    f2 = ta.ema("AAPL", 9)
    assert f2[-1] == f1[-1]
    assert ta.compute_count == 1
    # A different period is a separate compute.
    ta.ema("AAPL", 21)
    assert ta.compute_count == 2


def test_compute_happens_across_symbols_and_indicators():
    df = _df()
    ta, store = _ta(df, ("AAPL", "MSFT"))
    store.advance(df.index[60])
    ta.ema("AAPL", 9)
    ta.ema("AAPL", 21)
    ta.sma("AAPL", 20)
    ta.ema("MSFT", 9)
    assert ta.compute_count == 4  # (2 syms) with per-key caching


def test_cursor_narrows_visible_data_no_lookahead():
    df = _df()
    ta, store = _ta(df, ("AAPL",))
    # Indicator computed over the full series once, but reads clamp to cursor.
    ema20 = ta.sma("AAPL", 20)
    # Early cursor: visible length must equal the number of bars up to cursor.
    store.advance(df.index[10])
    assert len(ema20) == 11  # bars 0..10 inclusive
    # SMA(20) not yet defined with only 11 bars -> NaN at the cursor.
    assert np.isnan(ema20[-1])
    # A later cursor makes it defined, still without exposing future bars.
    store.advance(df.index[30])
    assert len(ema20) == 31
    assert not np.isnan(ema20[-1])
    assert ema20[-1] == pytest.approx(df[("AAPL", "close")].iloc[11:31].mean())
    # A future bar is NEVER reachable even by absolute positive index.
    assert ema20[1000] == ema20[-1]  # clamps to last visible, not future


def test_ohlcv_fields_are_cursor_truncated():
    df = _df()
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[5])
    o = ta.ohlcv("AAPL")
    assert o.count == 6
    assert o.close[-1] == pytest.approx(df[("AAPL", "close")].iloc[5])
    assert len(o.high) == 6
    assert len(o.volume) == 6


def test_atr_and_rolling_extremes():
    df = _df(60)
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[30])
    a = ta.atr("AAPL", 14)
    assert len(a) == 31
    assert a[-2] == a[-2]  # defined after warmup
    h = ta.highest("AAPL", 10)
    lo = ta.lowest("AAPL", 10)
    assert h[-1] == pytest.approx(df[("AAPL", "close")].iloc[21:31].max())
    assert lo[-1] == pytest.approx(df[("AAPL", "close")].iloc[21:31].min())


def test_rolling_sum_matches_reference():
    df = _df(40)
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[30])
    s = ta.sum("AAPL", 5)
    assert s[-1] == pytest.approx(df[("AAPL", "close")].iloc[26:31].sum())
    assert np.isnan(s[3])  # head before 5 bars is NaN


def test_adx_and_rsi_present():
    df = _df(80)
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[40])
    adx = ta.adx("AAPL", 14)
    rsi = ta.rsi("AAPL", 14)
    assert 0.0 <= adx[-1] <= 100.0
    assert 0.0 <= rsi[-1] <= 100.0


def test_unknown_interval_raises():
    df = _df()
    ta, store = _ta(df, ("AAPL",))
    with pytest.raises(ValueError):
        ta.ema("AAPL", 9, interval="4h")


def test_unknown_symbol_raises():
    df = _df()
    ta, store = _ta(df, ("AAPL",))
    with pytest.raises(KeyError):
        ta.ema("NOPE", 9)


def test_nan_head_is_preserved_then_converges():
    """SMA/EMA head (before `period` bars) is NaN; warm region is numeric."""
    df = _df(50)
    ta, store = _ta(df, ("AAPL",))
    store.advance(df.index[49])
    s = ta.sma("AAPL", 20)
    assert np.isnan(s[0])
    assert not np.isnan(s[-1])


def test_microbench_compute_is_per_key_not_per_candle():
    """Indicator compute stays O(1) per (sym, indicator, period): advancing the
    cursor across N candles must NOT trigger N recomputes -- only the first read
    of each key pays a full-series compute."""
    n = 120
    df = _df(n)
    ta, store = _ta(df, ("AAPL",))
    # Simulate a backtest: advance the cursor bar-by-bar and read the same two
    # EMAs each bar (as a DSL strategy would).
    for i in range(n):
        store.advance(df.index[i])
        f = ta.ema("AAPL", 9)
        s = ta.ema("AAPL", 21)
        if len(f) > 21:
            _ = f[-1] > s[-1]  # the per-candle read is O(1), cached
    # Exactly one full-series compute per (sym, period): 1 sym x 2 periods.
    assert ta.compute_count == 2


def test_init_ta_factory():
    df = _df()
    ta = init_ta(df, ("AAPL",), "1d")
    store = _store(df, ("AAPL",))
    ta.bind(store)
    store.advance(df.index[10])
    assert ta.close("AAPL")[-1] == pytest.approx(df[("AAPL", "close")].iloc[10])


def _df_with_gap(seed: int = 0) -> pd.DataFrame:
    """DataFrame where AAPL has a NaN-close gap at the middle index; MSFT is clean.

    Mirrors ``candle_generator`` semantics: a symbol with a NaN close has that
    candle skipped entirely (not accumulated) in the store.
    """
    df = _df(40, seed=seed)
    # Introduce a single-row gap (NaN close) for AAPL at index 20.
    for f in ("open", "high", "low", "close", "volume"):
        df.iloc[20, df.columns.get_loc(("AAPL", f))] = np.nan
    return df


def _gap_store(df: pd.DataFrame) -> CandleStore:
    """CandleStore whose rows mirror the generator: AAPL's NaN-close row is dropped."""
    rows: dict = {}
    for sym in ("AAPL",):
        mask = np.isfinite(df[("AAPL", "close")].to_numpy(dtype=float))
        n = int(mask.sum())
        arr = {
            k: np.empty(n) if k != "_len" else np.array([n])
            for k in ("timestamp", "open", "high", "low", "close", "volume", "_len")
        }
        arr["timestamp"] = df.index[mask].to_numpy().astype("datetime64[ms]")
        for f in ("open", "high", "low", "close", "volume"):
            arr[f] = df[(sym, f)].to_numpy(dtype=float)[mask]
        rows[(sym, "1d")] = arr
    return CandleStore(rows)


def test_gap_feed_stays_aligned_with_store():
    """Regression (#3): a NaN-close gap must not silently shift the feed's indices
    relative to the store. After a gap, ``view[-1]`` must return the *current*
    non-gap bar, not a stale/future one."""
    df = _df_with_gap()
    ta = TaContext.from_data(df, ("AAPL",), "1d")
    store = _gap_store(df)
    ta.bind(store)

    # The feed should hold exactly the non-gap rows (39 here: 40 - 1 skipped).
    assert len(ta.close("AAPL")) != len(df)  # gap dropped from feed

    # Cursor past the gap: visible length == store rows up to cursor, and the
    # last visible bar must be the real close at that timestamp, not a shifted one.
    # The gap was at df index 20; a cursor at df index 30 lands on store row 29
    # (row 20 skipped). The value must equal the non-gap close of df index 30.
    store.advance(df.index[30])
    assert store.cursor_count("AAPL", "1d") == 30  # 31 bars up to idx30 minus the gap
    assert len(ta.close("AAPL")) == 30
    assert ta.close("AAPL")[-1] == pytest.approx(df.iloc[30][("AAPL", "close")])

    # Cursor at the bar right before the gap: no gap has been reached, so indices
    # still align with the raw df (both feed and store count 0..19 -> 20 bars).
    store.advance(df.index[18])
    assert len(ta.close("AAPL")) == 19
    assert ta.close("AAPL")[-1] == pytest.approx(df.iloc[18][("AAPL", "close")])


def test_gap_misalignment_guard_fires_on_inconsistent_feed():
    """The access-time guard must fail loudly when a feed is *shorter* than the
    store's accumulated rows (a feed/storage mismatch that would serve a stale or
    misaligned bar). Simulates the pre-fix state: feed filtered (39 rows) but store
    kept the gap row (40 rows), so at the final cursor the store out-runs the feed."""
    df = _df_with_gap()
    ta = TaContext.from_data(df, ("AAPL",), "1d")  # properly filtered -> 39 rows
    store = _store(df, ("AAPL",))  # UNFILTERED store keeps all 40 rows
    ta.bind(store)
    # Cursor at the final bar: store counts 40 rows, feed only 39 -> guard fires.
    store.advance(df.index[-1])
    assert store.cursor_count("AAPL", "1d") == 40
    with pytest.raises(RuntimeError, match="misaligned"):
        _ = ta.close("AAPL")[-1]
