from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
from src.utils import get_ts

pf = Portfolio(
    PortfolioProps(
        stop_loss=0.10,
        take_profit=1.5,
        initial_capital=10000,
        position_size=0.1,
        commission=0.0001,
        start_date=get_ts("2025-01-01"),
    )
)


def test_on_signal():
    pass


def test_sl():
    pass


def test_tp():
    pass


def test_position_sizing():
    pass


def test_commissions():
    pass
