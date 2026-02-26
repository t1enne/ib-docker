from src.utils import get_ts
from typing import Optional
import click
import yaml
import asyncio

import src.mx as mx_mod
import src.spread as spread_mod
import src.nd as nd_mod
import src.pnd as pnd_mod
import src.syncm as sync_mod
import src.hmm as hmm_mod

from src.bt import StrategyType, backtest, load_strategy, StrategyConfig
from src.bt.metrics import get_backtest_results_analysis


@click.group()
def main():
    pass


@main.command(help="log correlation and cointegrations for the passed symbols")
@click.argument("symbols", nargs=-1)
@click.option("--universe", "-u", default=None, help="Path to universe config file")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--plot/--no-plot", default=False, help="Generate plotly heatmaps")
def mx(
    symbols: list[str],
    universe: Optional[str],
    start: Optional[str],
    end: Optional[str],
    plot: bool,
):
    s = get_ts(start) if start else None
    e = get_ts(end) if end else None
    mx_mod.matrix(symbols, s, e, plot, universe)


@main.command()
@click.argument("symbols", nargs=2)
@click.option("--start")
@click.option("--end")
@click.option("--rolling", "-r", type=int)
def spread(
    symbols: tuple[str, str],
    start: Optional[str],
    end: Optional[str],
    rolling: Optional[int] = None,
):
    s = get_ts(start) if start else None
    e = get_ts(end) if end else None
    spread_mod.spread(symbols, s, e, rolling)


@main.command(help="plot normalized deviation between price/returns and relative MA")
@click.argument("symbol")
@click.argument("ma", default=10)
def nd(symbol: str, ma: int):
    nd_mod.nd(symbol, ma)


@main.command(
    help="plot normalized deviation between pairs prices/returns vs their MAs"
)
@click.argument("symbols", nargs=2)
def pnd(symbols: list[str]):
    pnd_mod.pnd(symbols)


@main.command(help="analyze market regimes using Hidden Markov Model")
@click.argument("symbol")
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option(
    "--n-regimes", "-n", type=int, default=3, help="Number of regime states (2 or 3)"
)
@click.option(
    "--vol-window", "-v", type=int, default=20, help="Volatility calculation window"
)
@click.option(
    "--momentum-window", "-m", type=int, default=10, help="Momentum calculation window"
)
@click.option(
    "--min-train-size", type=int, default=252, help="Minimum observations for training"
)
@click.option("--update-interval", type=int, default=50, help="Retraining interval")
@click.option(
    "--output-dir",
    "-o",
    default="./hmm_models",
    help="Output directory for models and plots",
)
@click.option("--plot/--no-plot", default=True, help="Generate plots")
def hmm(
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    n_regimes: int,
    vol_window: int,
    momentum_window: int,
    min_train_size: int,
    update_interval: int,
    output_dir: str,
    plot: bool,
):
    s = get_ts(start) if start else None
    e = get_ts(end) if end else None
    hmm_mod.hmm(
        symbol,
        s,
        e,
        n_regimes,
        vol_window,
        momentum_window,
        min_train_size,
        update_interval,
        output_dir,
        plot,
    )


@main.command(help="run walk-forward backtest from strategy file")
@click.argument("strategy_file")
def bt(strategy_file: str):
    config = load_strategy(strategy_file)
    output = asyncio.run(backtest(config))
    click.echo(output)


@main.command(help="sync historical candle data for universe")
@click.option("--universe", default="universe.yml", help="Path to universe config file")
def sync(universe: str):
    data = sync_mod.load_universe_config(universe)
    asyncio.run(sync_mod.sync_data(data))


if __name__ == "__main__":
    main()
