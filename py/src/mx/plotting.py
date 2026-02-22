import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_matrices(
    corr_matrix: pd.DataFrame,
    cointegration_matrix: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Correlation Matrix", "Cointegration (p-values)"),
        horizontal_spacing=0.1,
    )

    corr_values = corr_matrix.values.astype(float)
    coin_values = cointegration_matrix.values.astype(float)

    fig.add_trace(
        go.Heatmap(
            z=corr_values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title="Correlation", len=0.4, y=0.8),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=coin_values,
            x=cointegration_matrix.columns,
            y=cointegration_matrix.index,
            colorscale="Greens",
            reversescale=True,
            zmin=0,
            zmax=1,
            text=coin_values,
            texttemplate="%{text:.3f}",
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title="p-value", len=0.4, y=0.2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"Correlation & Cointegration Analysis<br><sub>{min_date.date()} to {max_date.date()}</sub>",
            x=0.5,
            xanchor="center",
        ),
        height=500,
        width=1000,
        template="plotly_white",
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_yaxes(tickangle=0, row=1, col=1)
    fig.update_yaxes(tickangle=0, row=2, col=1)

    return fig
