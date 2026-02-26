from src.mx.plotting import plot_matrices
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yaml

from src.utils import get_log_returns, get_local_candles, symmetric_cointegration_p
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_metrics(df1: pd.DataFrame, df2: pd.DataFrame):
    corr = np.corrcoef(df1["Returns"], df2["Returns"])[0, 1]
    p_value = symmetric_cointegration_p(df1["Close"], df2["Close"])
    return round(corr, 2), round(p_value, 3)


def get_candles_info(data: Dict[str, pd.DataFrame]):
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    min_date = min(all_dates)
    max_date = max(all_dates)
    total_days = len(all_dates)
    return min_date, max_date, total_days


def matrix(
    input_symbols: list[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    plot: bool = False,
    universe: Optional[str] = None,
):
    if universe:
        with open(universe) as f:
            data = yaml.safe_load(f)
            input_symbols = data.get("universe", [])
    elif not input_symbols:
        raise ValueError("Provide either --universe/-u or positional symbols")

    input_symbols = list(map(lambda s: s.upper(), input_symbols))
    candles_by_sym = {
        sym: get_local_candles(sym.upper(), start, end, bar="1h")
        for sym in input_symbols
    }

    candlex_ref_amount = len(candles_by_sym[input_symbols[0]])
    symbols = []
    returns = {}
    for sym in input_symbols:
        is_empty = candles_by_sym[sym].empty
        is_len_consistent = len(candles_by_sym[sym]) == candlex_ref_amount
        if is_empty:
            print(f"No candles found for {sym}")
            continue

        if not is_len_consistent:
            print(f"Inconsistent len")
            continue

        returns[sym] = get_log_returns(candles_by_sym[sym])
        symbols.append(sym)

    # data = {sym: get_log_returns()}

    min, max, total = get_candles_info(returns)
    corr_matrix = pd.DataFrame(index=pd.Index(symbols), columns=pd.Index(symbols))
    cointegration_matrix = pd.DataFrame(
        index=pd.Index(symbols), columns=pd.Index(symbols)
    )

    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            if i == j:
                corr_matrix.loc[sym1, sym2] = 1.0
                cointegration_matrix.loc[sym1, sym2] = 0.0
            else:
                print(sym1, sym2)
                corr, p_val = get_metrics(returns[sym1], returns[sym2])
                corr_matrix.loc[sym1, sym2] = corr
                cointegration_matrix.loc[sym1, sym2] = p_val

    print(f"Date range: {min.date()} to {max.date()}, Total days: {total}")
    print("Correlation Matrix:")
    print(corr_matrix)
    print("\nCointegration Matrix (p-values):")
    print(cointegration_matrix)

    if not plot:
        return

    fig = plot_matrices(corr_matrix, cointegration_matrix, min, max)
    output_file = "plots/correlation_cointegration_heatmap.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")


__all__ = ["matrix"]
