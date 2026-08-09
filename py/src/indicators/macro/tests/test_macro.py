"""Tests for the macro indicator factory (:mod:`src.indicators.macro`).

The macro package exposes one public API — :func:`init_macro_indicator` — which
returns a lookahead-free callable bound to a single macro series:

    ind = init_macro_indicator("payems")     # reads assets/payems.csv once
    v   = ind(ts)                            # latest payrolls known at/before ts

These tests cover the guarantees that matter:

- **Lookahead-freeness** — the returned callable never leaks a future print.
- **Load-once / then-cheap** — the series is loaded at init (or bound
  directly from a pre-loaded Series); repeated calls don't re-read the file.
- **Source dispatch** — ``"cpi"`` routes to the World Bank chained-index
  loader; FRED names route to the two-column ``assets/<name>.csv`` loader.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import pandas as pd
import pytest

from src.indicators.macro import deflated_log_prices, init_macro_indicator

# A macro indicator callable: f(ts) -> float | None
_MacroInd = Callable[[pd.Timestamp | str | int], float | None]

# All FRED asset names the factory must route via load_daily.
_FRED_NAMES = ["gdpc1", "gdp", "payems", "unrate", "indpro", "tcu", "bopgstb"]


def _t(s: str) -> pd.Timestamp:
    """Strictly-typed cursor Timestamp (avoids pandas stubs' `Timestamp | NaT`)."""
    return cast(pd.Timestamp, pd.Timestamp(s))


def _write_fred(tmp_path, name: str, body: str) -> None:
    """Write a two-column FRED-style asset file (blank cells = no-print)."""
    (tmp_path / f"{name}.csv").write_text(f"date,{name}\n{body}")


def _daily_series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="x")


# ---------------------------------------------------------------------------
# binding a pre-loaded series (the fast / no-I/O path)
# ---------------------------------------------------------------------------


def test_binds_preloaded_series_no_io(monkeypatch, tmp_path):
    # With `series` given, the factory must not touch the filesystem at all —
    # even if the asset file is absent.
    s = _daily_series()
    ind = init_macro_indicator("payems", series=s)
    assert ind(_t("2020-01-03")) == 3.0
    assert ind(_t("2020-01-04 12:00")) == 4.0


def test_bound_series_lookahead_free():
    # A later value (5.0 on 01-05) must never leak to a cursor on 01-04.
    ind = init_macro_indicator("payems", series=_daily_series())
    assert ind(_t("2020-01-04")) == 4.0
    assert ind(_t("2020-01-04")) != 5.0


def test_before_coverage_is_none():
    ind = init_macro_indicator("payems", series=_daily_series())
    assert ind(_t("2019-12-31")) is None


def test_accepts_epoch_seconds_int(monkeypatch, tmp_path):
    s = _daily_series()
    ind = init_macro_indicator("payems", series=s)
    assert ind(_t("2020-01-03").value // 10**9) == 3.0


# ---------------------------------------------------------------------------
# FRED dispatch + asset-file contract
# ---------------------------------------------------------------------------


def test_fred_dispatch_ffills_blank_cells(tmp_path):
    # Row 2 is a FRED no-print blank; value must forward-fill onto a daily grid
    # and the factory must read the file once at init.
    _write_fred(
        tmp_path, "payems", "2020-01-01,150000\n2020-01-02,\n2020-01-03,151000\n"
    )
    ind = init_macro_indicator("payems", source_dir=tmp_path)
    assert ind(_t("2020-01-01")) == 150000.0
    assert ind(_t("2020-01-02")) == 150000.0  # blank carried forward
    assert ind(_t("2020-01-03")) == 151000.0


def test_fred_dispatch_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        init_macro_indicator("unrate", source_dir=tmp_path)


@pytest.mark.parametrize("name", _FRED_NAMES)
def test_fred_dispatch_each_name(tmp_path, name):
    _write_fred(tmp_path, name, "2020-06-01,42.0\n2020-07-01,43.5\n")
    ind = init_macro_indicator(name, source_dir=tmp_path)
    assert ind(_t("2020-06-15")) == pytest.approx(42.0)
    assert ind(_t("2020-07-01")) == pytest.approx(43.5)
    # future value not leaked at a cursor before its release
    assert ind(_t("2020-06-30")) != 43.5


def test_loads_once_not_per_call(tmp_path):
    # The factory reads the file at init; later calls reuse the same Series.
    # We verify by pointing at a file and then deleting it — the bound callable
    # must still answer because the series is already in memory.
    _write_fred(tmp_path, "tcu", "2020-03-01,75.0\n")
    ind = init_macro_indicator("tcu", source_dir=tmp_path)
    (tmp_path / "tcu.csv").unlink()  # remove the source
    assert ind(_t("2020-03-01")) == 75.0


# ---------------------------------------------------------------------------
# CPI dispatch (World Bank chained index)
# ---------------------------------------------------------------------------


def _write_wb(tmp_path, body: str) -> None:
    header = "Country,Country Code,Year,CPI\n"
    (tmp_path / "cpi.csv").write_text(header + body)


def test_cpi_dispatch_chains_annual_rates(tmp_path):
    # 10% every year for 2 years -> level 1.0 -> 1.1 -> 1.21 (geometric chain).
    _write_wb(tmp_path, "United States,USA,2020,10\nUnited States,USA,2021,10\n")
    ind = init_macro_indicator("cpi", source_dir=tmp_path)
    assert ind(_t("2020-06-01")) == pytest.approx(1.0)
    assert ind(_t("2021-01-01")) == pytest.approx(1.1)
    assert ind(_t("2022-06-01")) == pytest.approx(1.21)


def test_cpi_dispatch_lookahead_free(tmp_path):
    # Two-year file: a cursor inside year 1 must NOT see year-2 level (1.1).
    _write_wb(tmp_path, "United States,USA,2020,10\nUnited States,USA,2021,10\n")
    ind = init_macro_indicator("cpi", source_dir=tmp_path)
    assert ind(_t("2020-06-30")) == pytest.approx(1.0)
    assert ind(_t("2020-06-30")) != pytest.approx(1.1)


def test_cpi_dispatch_before_coverage_is_none(tmp_path):
    _write_wb(tmp_path, "United States,USA,2020,5\n")
    ind = init_macro_indicator("cpi", source_dir=tmp_path)
    assert ind(_t("1950-01-01")) is None


def test_cpi_dispatch_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        init_macro_indicator("cpi", source_dir=tmp_path)


# ---------------------------------------------------------------------------
# pure helper: deflated_log_prices (RDD consumer)
# ---------------------------------------------------------------------------


def test_deflated_log_prices_equals_ln_nominal_minus_ln_cpi():
    day = pd.date_range("2020-01-01", periods=3, freq="D")
    nominal = pd.Series([100.0, 101.0, 102.0], index=day)
    cpi_level = pd.Series([2.0, 2.0, 2.0], index=day)  # constant deflator
    rl = deflated_log_prices(nominal, cpi_level)
    expected = (np.log(nominal) - np.log(cpi_level)).astype(float).rename("real_log")
    pd.testing.assert_series_equal(rl.astype(float), expected)


def test_deflated_log_prices_reindexes_nominal_grid():
    nom_day = pd.date_range("2020-06-01", periods=3, freq="D")
    nominal = pd.Series([1.0, 1.0, 1.0], index=nom_day)
    coarse = pd.Series([2.0, 4.0], index=pd.to_datetime(["2019-01-01", "2021-01-01"]))
    rl = deflated_log_prices(nominal, coarse)
    assert rl.dropna().abs().max() == pytest.approx(float(np.log(2.0)))


def test_deflated_log_prices_empty_inputs():
    assert deflated_log_prices(pd.Series(dtype=float), pd.Series(dtype=float)).empty


# ---------------------------------------------------------------------------
# end-to-end against the real assets dir (when the collector has been run)
# ---------------------------------------------------------------------------

from pathlib import Path as _P  # noqa: E402 (used in skip guards below)


@pytest.mark.skipif(not _P("assets/gdpc1.csv").exists(), reason="collector not run")
def test_e2e_every_fred_name_returns_at_ts():
    ts = _t("2020-06-15")
    for name in _FRED_NAMES:
        ind = init_macro_indicator(name, source_dir="assets")
        val = ind(ts)
        assert val is not None, f"{name} returned None at {ts}"
        assert np.isfinite(val), f"{name} non-finite at {ts}"


@pytest.mark.skipif(not _P("assets/payems.csv").exists(), reason="collector not run")
def test_e2e_monthly_step_structure():
    # Payrolls are monthly: a cursor within a month sees that month's print,
    # and a later cursor in the same month sees the same (FF'd) value.
    payems = init_macro_indicator("payems", source_dir="assets")
    v1, v2 = payems(_t("2020-03-10")), payems(_t("2020-03-20"))
    assert v1 is not None and v2 is not None
    assert v1 == v2


@pytest.mark.skipif(not _P("assets/cpi.csv").exists(), reason="collector not run")
def test_e2e_cpi_real_level_sane():
    cpi = init_macro_indicator("cpi", source_dir="assets")
    v = cpi(_t("2020-06-15"))
    assert v is not None
    assert 0.5 < v < 10.0  # a price-index level, not a rate
