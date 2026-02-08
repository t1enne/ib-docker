from typing import Optional
import click
import yaml
import asyncio

import src.mx as mx_mod
import src.spread as spread_mod
import src.nd as nd_mod
import src.pnd as pnd_mod
import src.syncm as sync_mod

from src.bt import StrategyType, backtest, load_strategy, Strategy


@click.group()
def main():
    pass


@main.command(help="log correlation and cointegrations for the passed symbols")
@click.argument("symbols", nargs=-1)
def mx(symbols: list[str]):
    mx_mod.matrix(symbols)


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
    spread_mod.spread(symbols, start, end, rolling)


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


@main.command(help="create and save a strategy configuration")
@click.argument("strategy_type", type=click.Choice([s.value for s in StrategyType]))
@click.argument("symbols", nargs=-1)
@click.option("--name", help="Strategy name")
@click.option("--ma-period", default=20, help="Moving average period")
@click.option("--entry-z", default=2.0, help="Entry z-score")
@click.option("--stop-loss", default=0.05, help="Stop loss")
@click.option("--take-profit", default=0.10, help="Take profit")
@click.option("--initial-capital", default=100000, help="Initial capital")
@click.option("--position-size", default=0.1, help="Position size")
@click.option("--commission", default=0.001, help="Commission")
@click.option("--training-start", required=True, help="Training start date")
@click.option("--training-end", required=True, help="Training end date")
@click.option("--trading-start", required=True, help="Trading start date")
@click.option("--trading-end", required=True, help="Trading end date")
@click.option(
    "--retrain-tick-interval", default=1000, help="Retraining interval in ticks"
)
@click.option("--plot/--no-plot", default=True, help="Plot results")
@click.option("--output", "-o", help="Output YAML file")
def strategy(
    strategy_type: str,
    symbols: tuple[str],
    name: str,
    entry_z: float,
    exit_z: float,
    stop_loss: float,
    take_profit: float,
    initial_capital: float,
    position_size: float,
    commission: float,
    training_start: str,
    training_end: str,
    trading_start: str,
    trading_end: str,
    rolling_window_size: int,
    plot: bool,
    output: str,
):
    if not name:
        name = f"{strategy_type}_{'_'.join(symbols)}"
    if not output:
        output = f"{name}.yaml"
    strat = Strategy(
        name=name,
        strategy_type=strategy_type,
        symbols=list(symbols),
        entry_z=entry_z,
        exit_z=exit_z,
        stop_loss=stop_loss,
        take_profit=take_profit,
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission,
        training_start=training_start,
        training_end=training_end,
        trading_start=trading_start,
        trading_end=trading_end,
        rolling_window_size=rolling_window_size,
        plot=plot,
    )

    with open(output, "w") as f:
        yaml.dump(strat.__dict__, f)
    click.echo(f"Strategy saved to {output}")


@main.command(help="run walk-forward backtest from strategy file")
@click.argument("strategy_file")
def bt(strategy_file: str):
    asyncio.run(backtest(load_strategy(strategy_file)))


@main.command(help="sync historical candle data for universe")
@click.option("--universe", default="universe.yml", help="Path to universe config file")
def sync(universe: str):
    data = sync_mod.load_universe_config(universe)
    asyncio.run(sync_mod.sync_data(data))


if __name__ == "__main__":
    main()
