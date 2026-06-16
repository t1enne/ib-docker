from typing import Optional
import pandas as pd
from pandas import Timestamp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import get_local_candles, parse_timestamp
from src.kalman.pure import run_pairs_kalman
from src.kalman.types import PairsKalmanConfig, PairsKalmanResult


def spread(
    symbols: tuple[str, str],
    start_date: Optional[Timestamp] = None,
    end_date: Optional[Timestamp] = None,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
    mean_halflife: int = 50,
    bar: str = "1d",
):
    sym1, sym2 = symbols
    ma_windows = [50]
    _start = parse_timestamp(start_date) if start_date else None
    _end = parse_timestamp(end_date) if end_date else None
    df1 = get_local_candles(sym1.upper(), _start, _end, bar)
    df2 = get_local_candles(sym2.upper(), _start, _end, bar)

    ratio = pd.DataFrame(
        {
            "ratio": df1["close"] / df2["close"],
            f"{sym1}": df1["close"],
            f"{sym2}": df2["close"],
        },
        index=df1.index,
    )

    # Pairs Kalman filter: [α, β] two-state model
    cfg = PairsKalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mean_halflife=mean_halflife,
    )
    result = run_pairs_kalman(df1["close"], df2["close"], config=cfg)

    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=[
            f"t-Statistic (spread / √S) {sym1}-{sym2}",
            f"Hedge Ratio (β) {sym1}-{sym2}",
            f"Intercept (α) {sym1}-{sym2}",
            "Innovation Covariance S",
            "Price Ratio",
        ],
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
        row_heights=[0.25, 0.15, 0.15, 0.15, 0.3],
    )

    # Row 1 — t_stat (THE trading signal)
    fig.add_trace(
        go.Scatter(
            x=result.t_stat.index,
            y=result.t_stat,
            mode="lines",
            name=f"t-stat {sym1}-{sym2}",
            line=dict(color="purple", width=1.5),
        ),
        row=1,
        col=1,
    )
    for window in ma_windows:
        t_ema = result.t_stat.ewm(span=window).mean()
        fig.add_trace(
            go.Scatter(
                x=result.t_stat.index,
                y=t_ema,
                mode="lines",
                name=f"t-stat EMA {window}",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )
    fig.add_hline(y=2.0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Row 2 — beta
    fig.add_trace(
        go.Scatter(
            x=result.beta.index,
            y=result.beta,
            mode="lines",
            name="β",
            line=dict(color="seagreen", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=1)

    # Row 3 — alpha
    fig.add_trace(
        go.Scatter(
            x=result.alpha.index,
            y=result.alpha,
            mode="lines",
            name="α",
            line=dict(color="darkorange", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=3, col=1)

    # Row 4 — innovation covariance S
    fig.add_trace(
        go.Scatter(
            x=result.innovation_S.index,
            y=result.innovation_S,
            mode="lines",
            name="S (innov cov)",
            line=dict(color="crimson", width=1),
        ),
        row=4,
        col=1,
    )

    # Row 5 — ratio
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio["ratio"],
            mode="lines",
            name="ratio",
            hovertemplate=f"{sym1}: %{{customdata[0]:.2f}}<br>{sym2}: %{{customdata[1]:.2f}}<br>Ratio: %{{y:.4f}}<extra></extra>",
            customdata=ratio[[f"{sym1}", f"{sym2}"]].values,
        ),
        row=5,
        col=1,
    )
    for window in ma_windows:
        ratio_ema = ratio["ratio"].ewm(span=window).mean()
        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio_ema,
                mode="lines",
                name=f"ratio EMA {window}",
                line=dict(width=1),
            ),
            row=5,
            col=1,
        )

    fig.update_xaxes(type="date")

    output_file = f"plots/spread_{sym1}_{sym2}.html"
    fig.write_html(output_file)
    print(f"Saved to {output_file}")
