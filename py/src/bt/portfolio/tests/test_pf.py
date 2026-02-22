import pytest
from src.bt.portfolio import Portfolio, PortfolioProps
from src.bt.types import TradeSignal, ActionType, Tick, TradeExitReason, FillEvent
from src.utils import get_ts


@pytest.fixture
def pf():
    return Portfolio(
        PortfolioProps(
            stop_loss=0.1,
            take_profit=0.5,
            initial_capital=10000,
            position_size=0.1,
            commission=1,
            start_date=get_ts("2024-12-31"),
        )
    )


def get_fill(s: TradeSignal, pf: Portfolio):
    return FillEvent(
        signal=s,
        filled_qty=1,
        executed_price=s.price,
        commission=pf.commission,
        slippage=0.0,
    )


# ---------------------------------------------------------------------------
# Long-side tests (regression guards)
# ---------------------------------------------------------------------------


def test_long_open_cash(pf: Portfolio):
    """Opening a long should debit cash by notional + commission."""
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    cash_before = pf.cash
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None

    qty = trade.qty
    expected_cash = cash_before - (qty * 100.0) - pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_long_close_cash(pf: Portfolio):
    """Closing a long should credit cash by qty * exit_price - commission."""
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    pf.on_fill(get_fill(signal, pf))
    cash_after_open = pf.cash

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=110.0,
    )
    trade = pf.on_fill(get_fill(close_signal, pf))
    assert trade is not None

    qty = trade.qty
    expected_cash = cash_after_open + (qty * 110.0) - pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_long_round_trip_pnl(pf: Portfolio):
    """Long round-trip: final cash = initial - 2*commission + pnl."""
    entry_price, exit_price = 100.0, 120.0

    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    pf.on_fill(get_fill(close_signal, pf))

    pnl = (exit_price - entry_price) * qty
    expected_cash = pf.initial_capital + pnl - 2 * pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_long_equity_mark_to_market(pf: Portfolio):
    """Long equity = cash + qty * last_price."""
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=110.0,
        high=115.0,
        low=109.0,
        close=112.0,
        volume=1000,
    )
    pf.update_market_value(tick)

    expected_equity = pf.cash + qty * 112.0
    last_eq = pf.equity_curve["equity"].iloc[-1]
    assert abs(last_eq - expected_equity) < 0.01


# ---------------------------------------------------------------------------
# Short-side tests
# ---------------------------------------------------------------------------


def test_short_open_cash(pf: Portfolio):
    """Opening a short reserves collateral: cash -= notional + commission."""
    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    cash_before = pf.cash
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None

    qty = trade.qty
    expected_cash = cash_before - (qty * 100.0) - pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_short_close_winning(pf: Portfolio):
    """Winning short (price dropped): cash increases by collateral + profit - commission."""
    entry_price, exit_price = 100.0, 80.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty
    cash_after_open = pf.cash

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    pf.on_fill(get_fill(close_signal, pf))

    # On close: return collateral (qty * entry_price) and apply pnl, minus commission
    pnl = (entry_price - exit_price) * qty  # positive
    expected_cash = cash_after_open + (qty * entry_price) + pnl - pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_short_close_losing(pf: Portfolio):
    """Losing short (price rose): cash increases by collateral - loss - commission."""
    entry_price, exit_price = 100.0, 120.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty
    cash_after_open = pf.cash

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    pf.on_fill(get_fill(close_signal, pf))

    # On close: return collateral + pnl (negative here) - commission
    pnl = (entry_price - exit_price) * qty  # negative
    expected_cash = cash_after_open + (qty * entry_price) + pnl - pf.commission
    assert abs(pf.cash - expected_cash) < 0.01


def test_short_round_trip_pnl(pf: Portfolio):
    """Short round-trip: final cash = initial + pnl - 2*commission."""
    entry_price, exit_price = 100.0, 80.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    closed = pf.on_fill(get_fill(close_signal, pf))
    assert closed is not None

    pnl = (entry_price - exit_price) * qty
    expected_cash = pf.initial_capital + pnl - 2 * pf.commission
    assert abs(pf.cash - expected_cash) < 0.01
    assert abs(closed.pnl - pnl) < 0.01


def test_short_losing_round_trip(pf: Portfolio):
    """Losing short round-trip: final cash = initial - loss - 2*commission."""
    entry_price, exit_price = 100.0, 115.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=exit_price,
    )
    closed = pf.on_fill(get_fill(close_signal, pf))
    assert closed is not None

    pnl = (entry_price - exit_price) * qty  # negative
    expected_cash = pf.initial_capital + pnl - 2 * pf.commission
    assert abs(pf.cash - expected_cash) < 0.01
    assert pf.cash < pf.initial_capital  # lost money


# ---------------------------------------------------------------------------
# Short equity / mark-to-market tests
# ---------------------------------------------------------------------------


def test_short_equity_price_drops(pf: Portfolio):
    """Short equity when price drops: equity should increase (unrealized gain)."""
    entry_price = 100.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=96.0,
        low=88.0,
        close=90.0,
        volume=1000,
    )
    pf.update_market_value(tick)

    # Equity = initial_capital - open_commission + unrealized_pnl
    unrealized_pnl = qty * (entry_price - 90.0)
    expected_equity = pf.initial_capital - pf.commission + unrealized_pnl
    last_eq = pf.equity_curve["equity"].iloc[-1]
    assert abs(last_eq - expected_equity) < 0.01
    # Price dropped -> unrealized gain -> equity above what we started with (minus commission)
    assert last_eq > pf.initial_capital - pf.commission


def test_short_equity_price_rises(pf: Portfolio):
    """Short equity when price rises: equity should decrease (unrealized loss)."""
    entry_price = 100.0

    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=entry_price,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None
    qty = trade.qty

    tick = Tick(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=112.0,
        volume=1000,
    )
    pf.update_market_value(tick)

    # Equity = initial_capital - open_commission + unrealized_pnl (negative here)
    unrealized_pnl = qty * (entry_price - 112.0)
    expected_equity = pf.initial_capital - pf.commission + unrealized_pnl
    last_eq = pf.equity_curve["equity"].iloc[-1]
    assert abs(last_eq - expected_equity) < 0.01
    # Price rose -> unrealized loss -> equity below initial (minus commission)
    assert last_eq < pf.initial_capital - pf.commission


# ---------------------------------------------------------------------------
# SL/TP rounding consistency
# ---------------------------------------------------------------------------


def test_short_sl_tp_rounding(pf: Portfolio):
    """Short SL and TP should be rounded to 2 decimal places like longs."""
    signal = TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=-2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.33,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None

    # SL = 100.33 * (1 + 0.1) = 110.363 → should round to 110.36
    # TP = 100.33 * (1 - 0.5) = 50.165 → should round to 50.17 (or 50.16)
    assert trade.stop_loss == round(100.33 * 1.1, 2)
    assert trade.take_profit == round(100.33 * 0.5, 2)


def test_long_sl_tp_rounding(pf: Portfolio):
    """Long SL and TP should be rounded to 2 decimal places."""
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.33,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade is not None

    assert trade.stop_loss == round(100.33 * 0.9, 2)
    assert trade.take_profit == round(100.33 * 1.5, 2)


# ---------------------------------------------------------------------------
# Existing tests (preserved from original file)
# ---------------------------------------------------------------------------


def test_on_fill(pf: Portfolio):
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    trade = pf.on_fill(get_fill(signal, pf))

    assert trade is not None
    assert trade.symbol == "AAPL"
    assert trade.position == ActionType.long
    assert trade.entry_price == 100.0
    assert pf.positions["AAPL"] > 0
    assert pf.cash < 10000

    close_signal = TradeSignal(
        action=ActionType.close,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=110.0,
    )
    closed_trade = pf.on_fill(get_fill(close_signal, pf))
    assert closed_trade is not None
    assert closed_trade.exit_price == 110.0
    assert closed_trade.pnl > 0
    assert pf.positions["AAPL"] == 0


def test_close(pf: Portfolio):
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    trade = pf.on_fill(get_fill(signal, pf))
    assert trade

    qty = trade.qty

    s = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=0.0,
        timestamp=get_ts("2025-01-02"),
        price=120.0,
    )
    fe = FillEvent(
        signal=s,
        filled_qty=1,
        executed_price=s.price,
        commission=pf.commission,
        slippage=0,
    )
    pf._close_trade_from_fill(trade, fe)

    assert pf.cash > pf.initial_capital
