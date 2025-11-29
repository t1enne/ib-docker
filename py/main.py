import click

from src.mx import matrix


@click.group()
def main():
    pass


@main.command()
@click.argument("symbols", nargs=-1)
def mx(symbols: list[str]):
    matrix(symbols)


@main.command()
@click.argument("symbols", nargs=2)
def pair(symbols: tuple[str, str]):
    print(symbols)


if __name__ == "__main__":
    main()
