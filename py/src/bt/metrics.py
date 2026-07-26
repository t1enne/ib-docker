from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Any, cast
from scipy import stats

from src.bt.types import PortfolioResult, ActionType


@dataclass
class PerformanceMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    stability: float
    max_drawdown: float
    omega_ratio: float
    skewness: float
    kurtosis: float
    alpha: float
    beta: float


@dataclass
class DrawdownPeriod:
    peak_date: Optional[pd.Timestamp]
    valley_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    duration: int
    net_drawdown_pct: float


def _get_returns(equity_curve: pd.Series) -> pd.Series:
    returns = equity_curve.pct_change().dropna()
    return returns


def _equity_curve_time_info(equity_curve: pd.Series) -> tuple[float, float]:
    if len(equity_curve) < 2:
        return 0.0, 0.0

    index = equity_curve.index
    if isinstance(index, pd.DatetimeIndex):
        start = index[0]
        end = index[-1]
        try:
            elapsed_seconds = float((end - start).total_seconds())
        except AttributeError:
            elapsed_seconds = 0.0

        if elapsed_seconds > 0:
            elapsed_years = elapsed_seconds / (365.25 * 24 * 60 * 60)
            periods = len(equity_curve) - 1
            periods_per_year = periods / elapsed_years if elapsed_years > 0 else 0.0
            return elapsed_years, periods_per_year

    periods = len(equity_curve) - 1
    if periods <= 0:
        return 0.0, 0.0
    elapsed_years = periods / 252.0
    return elapsed_years, 252.0


def periods_per_year(equity_curve: pd.Series) -> float:
    _, ppy = _equity_curve_time_info(equity_curve)
    return ppy


def annual_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years, _ = _equity_curve_time_info(equity_curve)
    if years <= 0:
        return 0.0
    cagr = ((1 + total_return) ** (1 / years)) - 1
    return cagr


def annual_volatility(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0
    return float(returns.std() * np.sqrt(ppy))


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_periods(
    equity_curve: pd.Series, min_drawdown: float = 0.05
) -> List[DrawdownPeriod]:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    periods: List[DrawdownPeriod] = []
    in_drawdown = False
    peak_idx: Optional[pd.Timestamp] = None

    dates = list(drawdown.index)
    values = drawdown.values

    for i, (date, value) in enumerate(zip(dates, values)):
        if not in_drawdown and value < -min_drawdown:
            in_drawdown = True
            peak_idx = dates[0] if i == 0 else dates[i - 1]

        elif in_drawdown and value >= 0:
            in_drawdown = False
            if peak_idx is not None:
                valley_idx = i - 1
                valley_date = dates[valley_idx]
                recovery_date: Optional[pd.Timestamp] = date
                net_dd_pct = abs(values[valley_idx]) * 100

                try:
                    duration = (date - peak_idx).days
                except TypeError, AttributeError:
                    duration = i - dates.index(peak_idx) if peak_idx in dates else 0

                periods.append(
                    DrawdownPeriod(
                        peak_date=peak_idx,
                        valley_date=valley_date,
                        recovery_date=recovery_date,
                        duration=duration,
                        net_drawdown_pct=net_dd_pct,
                    )
                )
            peak_idx = None

    if in_drawdown and peak_idx is not None:
        valley_idx = np.argmin(values)
        valley_date = dates[valley_idx]
        net_dd_pct = abs(values[valley_idx]) * 100

        try:
            duration = (dates[-1] - peak_idx).days
        except TypeError, AttributeError:
            duration = len(dates) - dates.index(peak_idx) if peak_idx in dates else 0

        periods.append(
            DrawdownPeriod(
                peak_date=peak_idx,
                valley_date=valley_date,
                recovery_date=None,
                duration=duration,
                net_drawdown_pct=net_dd_pct,
            )
        )

    return sorted(periods, key=lambda x: x.net_drawdown_pct, reverse=True)


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    ret = annual_return(equity_curve)
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0
    downside_std_annual = downside_std * np.sqrt(ppy)
    return (ret - risk_free_rate) / downside_std_annual
    # return (ret - risk_free_rate) / downside_std


def omega_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    ppy = periods_per_year(equity_curve)
    per_period_rf = risk_free_rate / ppy if ppy > 0 else 0.0
    gain = float(returns[returns > per_period_rf].sum())
    loss = float(-returns[returns < per_period_rf].sum())
    if loss == 0:
        return 1.0
    return gain / loss


def skewness(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) < 3:
        return 0.0
    return float(returns.skew())


def kurtosis(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) < 4:
        return 0.0
    return float(returns.kurt())


def stability(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    x = np.arange(len(equity_curve))
    y = equity_curve.values
    result = stats.linregress(x, y)
    r_value: Any = getattr(result, "rvalue", None)
    if r_value is None:
        r_value = result[2] if isinstance(result, tuple) else 0.0
    return float(r_value) ** 2


def alpha_beta(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series | None = None,
) -> tuple[float, float]:
    returns = _get_returns(equity_curve)
    if len(returns) < 2:
        return 0.0, 1.0

    if benchmark_curve is not None:
        market_returns = _get_returns(benchmark_curve)
        # Align indices
        common = returns.index.intersection(market_returns.index)
        if len(common) < 2:
            return 0.0, 1.0
        returns = returns.loc[common]
        market_returns = market_returns.loc[common]
    else:
        # Fallback: use own returns as market proxy — weak, but backward-compat
        market_returns = returns

    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)

    if market_variance == 0:
        return 0.0, 1.0

    beta = covariance / market_variance
    period_alpha = float(returns.mean() - beta * market_returns.mean())
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0, float(beta)

    annualized_alpha = ((1 + period_alpha) ** ppy) - 1 if period_alpha > -1 else -1

    return annualized_alpha, float(beta)


def calmar_ratio(equity_curve: pd.Series) -> float:
    ret = annual_return(equity_curve)
    max_dd = abs(max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return ret / max_dd


def analyze_portfolio(result: PortfolioResult) -> PerformanceMetrics:
    """Project PortfolioResult → PerformanceMetrics (no recomputation)."""
    return PerformanceMetrics(
        annual_return=result.annual_return,
        annual_volatility=result.annual_volatility,
        sharpe_ratio=result.sharpe_ratio,
        calmar_ratio=result.calmar_ratio,
        sortino_ratio=result.sortino_ratio,
        stability=result.stability,
        max_drawdown=result.max_drawdown,
        omega_ratio=result.omega_ratio,
        skewness=result.skewness,
        kurtosis=result.kurtosis,
        alpha=result.alpha,
        beta=result.beta,
    )


def calculate_portfolio_result(
    equity_curve: pd.Series,
    trades,
    initial_capital: float,
    benchmark_curve: pd.Series | None = None,
) -> PortfolioResult:
    """Calculate portfolio result from equity curve and trades.

    Args:
        equity_curve: Equity curve as pandas Series
        trades: Iterable of Trade objects
        initial_capital: Starting capital
        benchmark_curve: Optional benchmark equity curve for alpha/beta

    Returns:
        PortfolioResult with all calculated metrics
    """
    returns = equity_curve.pct_change().dropna()
    ppy = periods_per_year(equity_curve)

    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    sharpe = 0.0
    if len(returns) > 0 and returns.std() != 0 and ppy > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(ppy)

    alpha, beta = alpha_beta(equity_curve, benchmark_curve)

    return PortfolioResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        trades=tuple(trades),
        equity_curve=equity_curve,
        annual_return=annual_return(equity_curve),
        annual_volatility=annual_volatility(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        calmar_ratio=calmar_ratio(equity_curve),
        sortino_ratio=sortino_ratio(equity_curve),
        omega_ratio=omega_ratio(equity_curve),
        skewness=skewness(equity_curve),
        kurtosis=kurtosis(equity_curve),
        stability=stability(equity_curve),
        alpha=alpha,
        beta=beta,
    )


def _safe_date_str(date: object | None) -> str:
    if date is None:
        return "NaT"
    if hasattr(date, "strftime"):
        return cast(Any, date).strftime("%Y-%m-%d")
    return str(date)[:10]


def get_backtest_results_analysis(
    result: PortfolioResult,
    metrics: Optional[PerformanceMetrics] = None,
    title: str = "Backtest Results",
    benchmark_curves: dict[str, pd.Series] | None = None,
) -> str:
    if metrics is None:
        metrics = analyze_portfolio(result)

    equity_curve = result.equity_curve
    trades = result.trades
    closed_trades = [t for t in trades if t.status.value == "closed"]
    profitable = [t for t in closed_trades if t.pnl > 0] if closed_trades else []
    win_rate = len(profitable) / len(closed_trades) if closed_trades else 0.0

    # -- Build lines --
    lines: list[str] = []

    lines.append(f"\n{title}")
    lines.append("=" * 80)

    # --- Benchmark Comparison ---
    if benchmark_curves:
        lines.append(f"\n{'Benchmark Comparison':<60}")
        sep = "-" * 72
        header = f"{'Benchmark':<12} {'Ann Ret':>10} {'Vol':>8} {'Sharpe':>8} {'Max DD':>8} {'Total Ret':>10}"
        lines.append(header)
        lines.append(sep)

        # Strategy row
        lines.append(
            f"{'Strategy':<12} {metrics.annual_return:>9.2%} "
            f"{metrics.annual_volatility:>7.2%} {metrics.sharpe_ratio:>7.2f} "
            f"{metrics.max_drawdown:>7.2%} {result.total_return:>9.2%}"
        )

        for bm_sym, bm_eq in benchmark_curves.items():
            bm_total = (bm_eq.iloc[-1] - bm_eq.iloc[0]) / bm_eq.iloc[0]
            bm_ann = annual_return(bm_eq)
            bm_vol = annual_volatility(bm_eq)
            bm_sharpe = 0.0
            if bm_vol > 0:
                bm_sharpe = bm_ann / bm_vol
            bm_dd = max_drawdown(bm_eq)

            lines.append(
                f"{bm_sym:<12} {bm_ann:>9.2%} "
                f"{bm_vol:>7.2%} {bm_sharpe:>7.2f} "
                f"{bm_dd:>7.2%} {bm_total:>9.2%}"
            )

        # Relative outperformance
        for bm_sym, bm_eq in benchmark_curves.items():
            bm_ann = annual_return(bm_eq)
            bm_dd = max_drawdown(bm_eq)
            excess_ret = metrics.annual_return - bm_ann
            lines.append(
                f"\n  vs {bm_sym}: alpha={metrics.alpha:+.2%}  "
                f"beta={metrics.beta:.2f}  "
                f"excess_ann_ret={excess_ret:+.2%}  "
                f"DD_improvement={abs(bm_dd) - abs(metrics.max_drawdown):+.2%}"
            )

    # --- Drawdowns ---
    dds = drawdown_periods(equity_curve)
    if dds:
        lines.append(f"\n{'Worst Drawdown Periods':<60}")
        header = f"{'Net DD %':>8}  {'Peak Date':>12}  {'Valley Date':>12}  {'Recovery':>12}  {'Days':>6}"
        lines.append(header)
        lines.append("-" * 62)
        for dd in dds[:5]:
            rec_str = (
                _safe_date_str(dd.recovery_date)
                if dd.recovery_date is not None
                else "  —"
            )
            lines.append(
                f"{dd.net_drawdown_pct:>7.2f}%  "
                f"{_safe_date_str(dd.peak_date):>12}  "
                f"{_safe_date_str(dd.valley_date):>12}  "
                f"{rec_str:>12}  "
                f"{dd.duration:>6}"
            )
        if not any(dds):
            lines.append("  (none)")
    else:
        lines.append("\nWorst Drawdown Periods")
        lines.append("  (none)")

    # --- Trades ---
    if trades:
        lines.append(f"\n{'Trades':<60}")
        # Columns: Sym  Entry         Exit          Entry$   Exit$    PnL$     Pos  Exit Reason
        sep = "-" * 114
        header = (
            f"{'Sym':<5} "
            f"{'Entry':>10}  "
            f"{'Exit':>10}  "
            f"{'Entry$':>8}  "
            f"{'Exit$':>8}  "
            f"{'PnL$':>9}  "
            f"{'Pos':>3}  "
            f"{'Exit Reason':<30}  "
            f"{'SL/TP':<12}"
        )
        lines.append(header)
        lines.append(sep)

        for t in trades:
            exit_price_str = (
                f"{t.exit_price:.2f}" if t.exit_price is not None else "     —"
            )
            pos_str = "L" if t.position == ActionType.long else "S"
            reason = str(t.close_reason)
            # Truncate long reasons
            if len(reason) > 30:
                reason = reason[:27] + "..."
            sl_tp = (
                f"{t.stop_loss:.2f}/{t.take_profit:.2f}"
                if t.stop_loss and t.take_profit
                else "—"
            )

            lines.append(
                f"{t.symbol:<5} "
                f"{_safe_date_str(t.entry_time):>10}  "
                f"{_safe_date_str(t.exit_time):>10}  "
                f"{t.entry_price:>7.2f}  "
                f"{exit_price_str:>8}  "
                f"{t.pnl:>8.2f}  "
                f"{pos_str:>3}  "
                f"{reason:<30}  "
                f"{sl_tp:<12}"
            )
    else:
        lines.append("\nTrades")
        lines.append("  (none)")

    # --- Trading Statistics ---
    lines.append(f"\n{'Trading Statistics':<25}")
    lines.append("-" * 42)
    lines.append(f"{'Starting Capital':<25} {equity_curve.iloc[0]:>15,.2f}")
    lines.append(f"{'Total Trades':<25} {len(trades):>15}")
    lines.append(f"{'Closed Trades':<25} {len(closed_trades):>15}")
    lines.append(f"{'Win Rate':<25} {win_rate:>14.2%}")
    lines.append(
        f"{'Total P&L':<25} {equity_curve.iloc[-1] - equity_curve.iloc[0]:>15,.2f}"
    )

    # --- Date Range ---
    first_str = _safe_date_str(equity_curve.index[0])
    last_str = _safe_date_str(equity_curve.index[-1])
    lines.append(f"\nData: {first_str} → {last_str}")

    # --- Duration ---
    n_periods = len(equity_curve)
    duration_parts = [f"{n_periods} periods"]
    if isinstance(equity_curve.index, pd.DatetimeIndex) and n_periods > 1:
        try:
            elapsed_days = (
                equity_curve.index[-1] - equity_curve.index[0]
            ).total_seconds() / 86400
        except AttributeError:
            elapsed_days = 0.0
        if elapsed_days > 0:
            months = elapsed_days / 30.44
            if months >= 12:
                duration_parts.append(f"{months / 12:.1f} years")
            else:
                duration_parts.append(f"{months:.1f} months")
    lines.append(f"Duration: {' · '.join(duration_parts)}")

    # --- Metrics Table ---
    lines.append(f"\n{'Metric':<22} {'Value':>10}")
    lines.append("-" * 34)
    lines.append(f"{'Annual Return':<22} {metrics.annual_return:>9.2%}")
    lines.append(f"{'Annual Volatility':<22} {metrics.annual_volatility:>9.2%}")
    lines.append(f"{'Sharpe Ratio':<22} {metrics.sharpe_ratio:>10.2f}")
    lines.append(f"{'Calmar Ratio':<22} {metrics.calmar_ratio:>10.2f}")
    lines.append(f"{'Sortino Ratio':<22} {metrics.sortino_ratio:>10.2f}")
    lines.append(f"{'Omega Ratio':<22} {metrics.omega_ratio:>10.2f}")
    lines.append(f"{'Max Drawdown':<22} {metrics.max_drawdown:>9.2%}")
    lines.append(f"{'Stability':<22} {metrics.stability:>10.2f}")
    lines.append(f"{'Skewness':<22} {metrics.skewness:>10.2f}")
    lines.append(f"{'Kurtosis':<22} {metrics.kurtosis:>10.2f}")
    lines.append(f"{'Alpha':<22} {metrics.alpha:>10.2f}")
    lines.append(f"{'Beta':<22} {metrics.beta:>10.2f}")

    return "\n".join(lines)


def exposure_and_turnover(
    result: PortfolioResult,
) -> dict[str, float]:
    trades = result.trades
    closed_trades = [t for t in trades if t.status.value == "closed"]

    if closed_trades:
        turnover = sum(abs(t.qty * t.entry_price) for t in closed_trades)
        ppy = periods_per_year(result.equity_curve)
        num_periods = len(result.equity_curve) - 1
        annualized_turnover = (
            turnover * ppy / num_periods if num_periods > 0 and ppy > 0 else turnover
        )
    else:
        turnover = 0.0
        annualized_turnover = 0.0

    return {
        "gross_exposure": turnover,
        "turnover": turnover,
        "annualized_turnover": float(annualized_turnover),
    }
