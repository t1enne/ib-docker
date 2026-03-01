import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, cast
from src.bt.types import (
    ActionType,
    PortfolioResult,
    StrategyConfig,
    BacktestResults,
    PlotConfig,
)


@dataclass(frozen=True)
class PlotMeta:
    num_rows: int
    titles: list[str]
    specs: list[list[dict | None]]


@dataclass(frozen=True)
class TradeMarkers:
    long_entries: list[tuple[pd.Timestamp, float]]
    long_entries_text: list[str]
    long_exits: list[tuple[pd.Timestamp, float]]
    long_exits_text: list[str]
    short_entries: list[tuple[pd.Timestamp, float]]
    short_entries_text: list[str]
    short_exits: list[tuple[pd.Timestamp, float]]
    short_exits_text: list[str]


def plot_backtest_results(
    config: StrategyConfig,
    bt_results: BacktestResults,
):
    z_scores = bt_results.z_scores
    regime_df = bt_results.regimes
    data = bt_results.data
    results = bt_results.pf
    symbols = config.symbols
    entry_z = config.strategy_params.get("entry_z", 0.0)
    exit_z = config.strategy_params.get("exit_z", 0.0)
    strategy = config.strategy_type
    plot_config = bt_results.plot_config

    has_z_scores = _has_data(z_scores)
    has_regime = _has_data(regime_df)
    has_strategy_subplots = bool(plot_config and len(plot_config.subplots) > 0)

    meta = _build_plot_meta(
        symbols,
        has_z_scores,
        has_regime,
        has_strategy_subplots,
        entry_z,
        exit_z,
        strategy,
    )
    fig = make_subplots(
        rows=meta.num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=meta.titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
        specs=meta.specs,
    )

    for i, symbol in enumerate(symbols):
        row = i + 1
        _add_price_and_volume(fig, row, symbol, data[symbol])

        # Add price overlays from strategy
        if plot_config and symbol in plot_config.price_overlays:
            for name, series in plot_config.price_overlays[symbol].items():
                _add_overlay(fig, row, series, name)

        markers = _collect_trade_markers(results, symbol)
        _add_trade_markers(fig, row, markers)
        fig.update_yaxes(title_text="Price", row=row, col=1, secondary_y=False)

    row_offset = len(symbols)

    # Add strategy-defined subplots (replaces hardcoded z-score)
    if has_strategy_subplots:
        assert plot_config is not None
        for subplot_title, series in plot_config.subplots:
            _add_series_subplot(fig, row_offset + 1, series, subplot_title)
            row_offset += 1
    elif has_z_scores:
        assert z_scores is not None
        _add_zscore_subplot(fig, row_offset + 1, z_scores, entry_z, exit_z)
        row_offset += 1

    if has_regime:
        assert regime_df is not None
        _add_regime_subplot(fig, row_offset + 1, regime_df)
        row_offset += 1

    equity_series = _equity_series(results)

    _add_equity_subplot(fig, row_offset + 1, equity_series)
    _add_drawdown_subplot(fig, row_offset + 2, equity_series)

    fig.update_layout(
        height=300 * meta.num_rows,
        title_text=f"Backtest Results - {strategy.upper()} Strategy",
        showlegend=False,
    )
    fig.update_xaxes(type="date")

    output_file = f"backtest_results_{strategy.lower().replace(' ', '_')}.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")


def _has_data(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and not df.empty


def _build_plot_meta(
    symbols: list[str],
    has_z_scores: bool,
    has_regime: bool,
    has_strategy_subplots: bool,
    entry_z: float,
    exit_z: float,
    strategy: str,
) -> PlotMeta:
    num_rows = (
        len(symbols)
        + (1 if has_strategy_subplots else (1 if has_z_scores else 0))
        + (1 if has_regime else 0)
        + 1
        + 1
    )
    titles = [f"{symbol} Price + Volume" for symbol in symbols]
    if has_strategy_subplots:
        pass  # Titles will be added by strategy
    elif has_z_scores:
        titles.append(f"Z-Score (Entry: ±{entry_z}, Exit: ±{exit_z})")
    if has_regime:
        titles.append("HMM Regime Probabilities")
    titles.extend([f"Equity Curve - {strategy.upper()} Strategy", "Drawdown"])

    specs: list[list[dict | None]] = []
    specs.extend([[{"secondary_y": True}] for _ in symbols])
    if has_strategy_subplots:
        pass  # Specs will be added by strategy
    elif has_z_scores:
        specs.append([{}])
    if has_regime:
        specs.append([{}])
    specs.append([{}])
    specs.append([{}])

    return PlotMeta(num_rows=num_rows, titles=titles, specs=specs)


def _add_price_and_volume(
    fig: go.Figure,
    row: int,
    symbol: str,
    df: pd.DataFrame,
) -> None:
    price_data = df["close"]
    volume_data = df["volume"]

    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data.values,
            mode="lines",
            name=f"{symbol} Price",
            line=dict(color="black", width=2),
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=volume_data.index,
            y=volume_data.values,
            name=f"{symbol} Volume",
            marker_color="rgba(100, 100, 200, 0.5)",
            showlegend=False,
        ),
        row=row,
        col=1,
        secondary_y=True,
    )

    max_vol = volume_data.max()
    if max_vol > 0:
        fig.update_yaxes(
            range=[0, max_vol * 4.5],
            showticklabels=False,
            showgrid=False,
            row=row,
            col=1,
            secondary_y=True,
        )


def _collect_trade_markers(results: PortfolioResult, symbol: str) -> TradeMarkers:
    long_entries: list[tuple[pd.Timestamp, float]] = []
    long_entries_text: list[str] = []
    long_exits: list[tuple[pd.Timestamp, float]] = []
    long_exits_text: list[str] = []
    short_entries: list[tuple[pd.Timestamp, float]] = []
    short_entries_text: list[str] = []
    short_exits: list[tuple[pd.Timestamp, float]] = []
    short_exits_text: list[str] = []

    for trade in results.trades:
        if trade.symbol != symbol:
            continue

        if trade.position == ActionType.long:
            long_entries.append((trade.entry_time, trade.entry_price))
            long_entries_text.append(f"Z: {trade.z_score:.2f}")
            if trade.exit_time and trade.exit_price is not None:
                long_exits.append((trade.exit_time, trade.exit_price))
                long_exits_text.append(
                    f"Reason: {trade.close_reason or 'unknown'}, PnL: {trade.pnl:.2f}"
                )
        else:
            short_entries.append((trade.entry_time, trade.entry_price))
            short_entries_text.append(f"Z: {trade.z_score:.2f}")
            if trade.exit_time and trade.exit_price is not None:
                short_exits.append((trade.exit_time, trade.exit_price))
                short_exits_text.append(
                    f"Reason: {trade.close_reason or 'unknown'}, PnL: {trade.pnl:.2f}"
                )

    return TradeMarkers(
        long_entries=long_entries,
        long_entries_text=long_entries_text,
        long_exits=long_exits,
        long_exits_text=long_exits_text,
        short_entries=short_entries,
        short_entries_text=short_entries_text,
        short_exits=short_exits,
        short_exits_text=short_exits_text,
    )


def _add_trade_markers(fig: go.Figure, row: int, markers: TradeMarkers) -> None:
    _add_marker_trace(
        fig,
        row,
        markers.long_entries,
        markers.long_entries_text,
        "Long Entry",
        symbol="triangle-up",
        color="green",
        text_position="top center",
    )
    _add_marker_trace(
        fig,
        row,
        markers.long_exits,
        markers.long_exits_text,
        "Long Exit",
        symbol="triangle-down",
        color="red",
        text_position="bottom center",
    )
    _add_marker_trace(
        fig,
        row,
        markers.short_entries,
        markers.short_entries_text,
        "Short Entry",
        symbol="triangle-down",
        color="yellow",
        text_position="top center",
    )
    _add_marker_trace(
        fig,
        row,
        markers.short_exits,
        markers.short_exits_text,
        "Short Exit",
        symbol="triangle-up",
        color="blue",
        text_position="bottom center",
    )


def _add_marker_trace(
    fig: go.Figure,
    row: int,
    points: Iterable[tuple[pd.Timestamp, float]],
    text: list[str],
    name: str,
    symbol: str,
    color: str,
    text_position: str,
) -> None:
    points_list = list(points)
    if not points_list:
        return

    times, prices = zip(*points_list)
    fig.add_trace(
        go.Scatter(
            x=list(pd.DatetimeIndex(times)),
            y=list(prices),
            mode="markers",
            name=name,
            marker=dict(symbol=symbol, color=color, size=8),
            text=text,
            textposition=text_position,
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def _add_overlay(fig: go.Figure, row: int, series: pd.Series, name: str):
    """Add a series as an overlay on the price chart."""
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            showlegend=False,
            opacity=0.5,
        ),
        row=row,
        col=1,
    )


def _add_series_subplot(fig: go.Figure, row: int, series: pd.Series, title: str):
    """Add a series as a separate subplot."""
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=title,
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text=title, row=row, col=1)


def _add_zscore_subplot(
    fig: go.Figure,
    row_z: int,
    z_scores: pd.DataFrame,
    entry_z: float,
    exit_z: float,
):
    fig.add_trace(
        go.Scatter(
            x=z_scores.index,
            y=z_scores["z"].values,
            mode="lines",
            name="Z-Score",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=row_z,
        col=1,
    )
    fig.add_hline(
        y=entry_z,
        line_dash="solid",
        line_color="green",
        row=cast(Any, row_z),
        col=cast(Any, 1),
        annotation_text=f"+{entry_z} entry",
    )
    fig.add_hline(
        y=-entry_z,
        line_dash="solid",
        line_color="red",
        row=cast(Any, row_z),
        col=cast(Any, 1),
        annotation_text=f"-{entry_z} entry",
    )
    fig.add_hline(
        y=exit_z,
        line_dash="dash",
        line_color="lightgreen",
        row=cast(Any, row_z),
        col=cast(Any, 1),
        annotation_text=f"+{exit_z} exit",
    )
    fig.add_hline(
        y=-exit_z,
        line_dash="dash",
        line_color="lightcoral",
        row=cast(Any, row_z),
        col=cast(Any, 1),
        annotation_text=f"-{exit_z} exit",
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        row=cast(Any, row_z),
        col=cast(Any, 1),
    )
    fig.update_yaxes(title_text="Z-Score", row=row_z, col=1)


def _add_regime_subplot(
    fig: go.Figure,
    row_offset: int,
    regime_df: pd.DataFrame,
) -> int:
    row_hmm = row_offset + 1

    colors = {
        0: "rgba(76, 175, 80, 0.5)",
        1: "rgba(255, 193, 7, 0.5)",
        2: "rgba(244, 67, 54, 0.5)",
    }
    labels = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}

    for i in [2, 1, 0]:
        col = f"prob_{i}"
        if col in regime_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=regime_df.index,
                    y=regime_df[col].values,
                    name=labels[i],
                    fill="tonexty" if i < 2 else "tozeroy",
                    line=dict(width=0.5, color=colors[i].replace("0.5", "1")),
                    fillcolor=colors[i],
                    showlegend=False,
                ),
                row=row_hmm,
                col=1,
            )

    _annotate_regime_transitions(fig, row_hmm, regime_df, labels)

    fig.update_yaxes(title_text="Regime Prob", range=[0, 1], row=row_hmm, col=1)
    return row_hmm


def _annotate_regime_transitions(
    fig: go.Figure,
    row: int,
    regime_df: pd.DataFrame,
    labels: Dict[int, str],
) -> None:
    regimes = regime_df["regime"].dropna()
    prev_regime = None
    for ts, regime in regimes.items():
        if regime != prev_regime and prev_regime is not None:
            prob_col = f"prob_{int(regime)}"
            prob = (
                regime_df.loc[ts, prob_col] if prob_col in regime_df.columns else None
            )
            prob_text = f" ({prob:.0%})" if prob is not None else ""
            fig.add_annotation(
                x=ts,
                y=1.0,
                text=f"{labels.get(int(regime), '?')}{prob_text}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-25,
                font=dict(size=9),
                row=row,
                col=1,
            )
        prev_regime = regime


def _equity_series(results: PortfolioResult) -> pd.Series:
    equity_data = results.equity_curve
    if isinstance(equity_data, dict):
        return pd.Series(equity_data)
    return equity_data


def _add_equity_subplot(fig: go.Figure, row: int, equity_series: pd.Series) -> None:
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode="lines",
            name="Equity Curve",
            line=dict(color="green", width=2),
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=row, col=1)


def _add_drawdown_subplot(fig: go.Figure, row: int, equity_series: pd.Series) -> None:
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            line=dict(color="red"),
            fill="tozeroy",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="Drawdown", row=row, col=1)
    fig.update_xaxes(title_text="Date", row=row, col=1)
