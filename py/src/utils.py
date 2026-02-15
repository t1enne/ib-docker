from typing import Any, Dict, List, Optional
import pandas as pd
import sqlite3
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime


def read_candles(
    symbol: str,
    start_date: Optional[str] = "2020-01-01",
    end_date: Optional[str] = "2099-01-01",
    bar: str = "1h",
) -> pd.DataFrame:
    # return pd.read_csv(f"mdata/{symbol}.csv")
    con = sqlite3.connect("../data/db.sqlite")
    print(con)
    cur = con.cursor()
    _sd = start_date and _parse_date(start_date).timestamp() * 1000 or ""
    _ed = end_date and _parse_date(end_date).timestamp() * 1000 or ""
    print(start_date, _sd)
    from_filter = f"and o.timestamp >= {int(_sd)}" if _sd else ""
    to_filter = f"and o.timestamp <= {int(_ed)}" if _ed else ""
    q = f"""select s.ticker as symbol,
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            from ohlcv_{bar} o left join symbol s
            on o.symbol_id = s.id
            where s.ticker = UPPER('{symbol}')
            {from_filter}
            {to_filter}
            """
    print(q)
    res = cur.execute(q)
    data = res.fetchall()
    con.close()
    columns = pd.Index(
        [
            "Symbol",
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )
    df = pd.DataFrame(data, columns=columns)
    df = df.assign(Date=pd.to_datetime(df["Timestamp"], unit="ms"))
    df = df.set_index("Date").drop(columns=["Timestamp"])
    print(df)
    return df


def _parse_date(date_str: str) -> datetime:
    """Parse date string, handling both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS formats."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")


def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "Returns"] = np.log(df["Close"] / df["Close"].shift(1)).round(4)
    return df.dropna()


def get_ols_fit_model(y, x):
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    if len(y_arr) == 0 or len(x_arr) == 0 or len(y_arr) != len(x_arr):
        raise ValueError("Empty or mismatched data for OLS")
    y_arr = np.log(y_arr).astype(float)
    x_arr = np.log(x_arr).astype(float)
    X = sm.add_constant(x_arr)
    return sm.OLS(y_arr, X).fit()


def hedge_ratio_residuals(y, x) -> pd.DataFrame:
    """Returns residuals from OLS(y ~ x) using log prices."""
    model = get_ols_fit_model(y, x)
    alpha, beta = model.params
    return y - (alpha + beta * x)


def _calculate_rolling_zscore_spread(s1: pd.Series, s2: pd.Series, rolling_window: int):
    if len(s1) < rolling_window:
        return pd.Series(dtype=float)

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


def eg_pvalue(price1, price2):
    """Engle-Granger p-value in one direction: price1 ~ price2"""
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
