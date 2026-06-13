from src.market_data import resample_ohlcv
from pandas.core.arrays import ExtensionArray
from typing import Any, Dict, List, Optional, Union, cast
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, date
from pathlib import Path
import os


_DEFAULT_START = pd.Timestamp("2020-01-01")
_DEFAULT_END = pd.Timestamp("2099-01-01")


def get_local_candles(
    symbol: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    bar: str = "1h",
) -> pd.DataFrame:
    # return pd.read_csv(f"mdata/{symbol}.csv")
    con = sqlite3.connect("../data/db.sqlite")
    cur = con.cursor()
    _start = start_date if start_date else _DEFAULT_START
    _end = end_date if end_date else _DEFAULT_END
    _start = cast(pd.Timestamp, _start)
    _end = cast(pd.Timestamp, _end)
    _sd = int(_start.timestamp() * 1000)
    _ed = int(_end.timestamp() * 1000)
    from_filter = f"and c.timestamp >= {_sd}" if _sd else ""
    to_filter = f"and c.timestamp <= {_ed}" if _ed else ""
    q = f"""select s.ticker as symbol,
                c.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume
            from candle c left join symbol s
            on c.conid = s.conid
            where s.ticker = UPPER('{symbol}')
            {from_filter}
            {to_filter}
            """
    res = cur.execute(q)
    data = res.fetchall()
    con.close()
    columns = pd.Index(
        [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )
    df = pd.DataFrame(data, columns=columns)
    df = df.assign(Date=pd.to_datetime(df["timestamp"], unit="ms"))
    df = df.set_index("Date").drop(columns=["timestamp"])
    if bar != "1h":
        return resample_ohlcv(df, bar, completed_only=True)

    return df


def _parse_date(date_str: str) -> datetime:
    """Parse date string, handling both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS formats."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")


def get_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "Returns"] = np.log(df["close"] / df["close"].shift(1)).round(4)
    return df.dropna()


def _calculate_rolling_zscore_spread(s1: pd.Series, s2: pd.Series, rolling_window: int):
    if len(s1) < rolling_window:
        return pd.Series(dtype=float)

    import statsmodels.api as sm  # lazy import

    # Calculate rolling spread using OLS
    def calc_spread(window_idx):
        idx = s1.index[window_idx]
        if window_idx < rolling_window - 1:
            return np.nan
        start_idx = window_idx - rolling_window + 1
        end_idx = window_idx + 1

        s1_window = s1.iloc[start_idx:end_idx]
        s2_window = s2.iloc[start_idx:end_idx]

        if s1_window.empty or s2_window.empty:
            return np.nan

        X = sm.add_constant(s2_window)
        model = sm.OLS(s1_window, X).fit()
        alpha, beta = model.params
        return s1.loc[idx] - (alpha + beta * s2.loc[idx])

    # Calculate rolling spread
    spread_series = pd.Series([calc_spread(i) for i in range(len(s1))], index=s1.index)

    # Calculate rolling z-score
    rolling_mean = spread_series.rolling(
        window=rolling_window, min_periods=rolling_window
    ).mean()
    rolling_std = spread_series.rolling(
        window=rolling_window, min_periods=rolling_window
    ).std()

    return (spread_series - rolling_mean) / rolling_std


def get_ols_fit_model(y, x):
    import statsmodels.api as sm  # lazy import

    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    if len(y_arr) == 0 or len(x_arr) == 0 or len(y_arr) != len(x_arr):
        raise ValueError("Empty or mismatched data for OLS")
    y_arr = np.log(y_arr).astype(float)
    x_arr = np.log(x_arr).astype(float)
    X = sm.add_constant(x_arr)
    return sm.OLS(y_arr, X).fit()


def calculate_zscore_spread(
    s1: pd.Series, s2: pd.Series, rolling_window: Optional[int] = None
):
    """Calculate z-score normalized spread"""
    if s1.empty or s2.empty or len(s1) != len(s2):
        return pd.Series(dtype=float)
    if rolling_window is not None and rolling_window > 0:
        return _calculate_rolling_zscore_spread(s1, s2, rolling_window)

    model = get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    scaled_s2 = alpha + beta * s2
    spread_series = s1 - scaled_s2
    if spread_series.empty:
        return pd.Series(dtype=float)
    return (spread_series - spread_series.mean()) / spread_series.std()


def hedge_ratio_residuals(y, x) -> pd.DataFrame:
    """Returns residuals from OLS(y ~ x) using log prices."""
    model = get_ols_fit_model(y, x)
    alpha, beta = model.params
    return y - (alpha + beta * x)


def eg_pvalue(price1, price2):
    """Engle-Granger p-value in one direction: price1 ~ price2"""
    from statsmodels.tsa.stattools import adfuller  # lazy import

    resid = hedge_ratio_residuals(price1, price2)
    return adfuller(resid)[1]  # p-value


def symmetric_cointegration_p(price1, price2):
    """
    Proper symmetric cointegration:
    - compute p-values in both directions
    - take the minimum — the statistically valid symmetric measure
    """
    p1 = float(eg_pvalue(price1, price2))
    p2 = float(eg_pvalue(price2, price1))
    # return min(p1, p2)  # recommended
    return float((p1 + p2) / 2)


def pick(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict[str, Any], keys: List[str]):
    return {k: v for k, v in d.items() if k not in keys}


def validate_schema(data, schema):
    """
    Validate complex nested data structures
    """

    def check_type(value, expected_type):
        if isinstance(expected_type, type):
            return isinstance(value, expected_type)

        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                return False
            return all(
                key in value and check_type(value[key], type_check)
                for key, type_check in expected_type.items()
            )

        return False

    return check_type(data, schema)


def get_ts(ds: str) -> pd.Timestamp:
    _ts = pd.Timestamp(ds)
    if isinstance(_ts, pd.Timestamp):
        return _ts

    raise ValueError(f"Failed when creating timestamp for {ds}")


def parse_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError(f"Invalid timestamp: {value}")
        return value
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value}")
    return cast(pd.Timestamp, timestamp)


def get_days_from_now(from_date: date) -> int:
    now = date.today()
    diff = now - from_date
    return diff.days


def list_to_axes(l: list[str]) -> ExtensionArray:
    return cast(ExtensionArray, l)


def load_env(path: str | Path = ".env") -> None:
    """Load a .env file into os.environ. Never overrides existing env vars."""

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = val
