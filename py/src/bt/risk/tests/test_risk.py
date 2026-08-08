"""Tests for risk checks — stop loss, take profit, trailing stop."""

import pytest
from src.bt.risk import (
    check_risk,
    check_position_risk,
    update_trailing_stop,
    create_risk_config,
)
from src.bt.state import (
    Position,
    Candle,
    ActionType,
    StopLossEvent,
    TakeProfitEvent,
    RiskConfig,
)
from src.utils import get_ts


@pytest.fixture
def risk_config():
    return create_risk_config(stop_loss_pct=0.1, take_profit_pct=0.2)


# ── check_position_risk (returns (Position, event|None)) ──────────────


def test_stop_loss_long(risk_config):
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=95.0,
        high=98.0,
        low=85.0,
        close=88.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, risk_config)
    assert isinstance(event, StopLossEvent)
    assert event.symbol == "AAPL"
    assert event.trigger_price == 90.0
    # Explicit SL — no change to stop level
    assert new_pos.stop_loss == 90.0


def test_take_profit_short(risk_config):
    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=110.0,
        take_profit=80.0,
        last_price=100.0,
        type=ActionType.short,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=85.0,
        high=90.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, risk_config)
    assert isinstance(event, TakeProfitEvent)
    assert event.trigger_price == 80.0


def test_no_trigger_when_inside_range(risk_config):
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, risk_config)
    assert event is None


# ── Explicit vs auto-managed SL/TP ────────────────────────────────────


def test_strategy_sl_not_trailed_when_trailing_disabled():
    """Strategy-set SL is fixed unless trailing_stop is explicitly enabled.
    The default engine config has trailing_stop=False, so a strategy-provided
    SL must never move."""
    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, config)
    # SL unchanged — trailing disabled by default
    assert new_pos.stop_loss == 90.0
    assert event is None


def test_no_sl_tp_without_explicit_levels_long():
    """No config-level SL/TP fallback: a position with no levels stays level-less."""
    config = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.15, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, config)
    assert event is None
    # No strategy-provided levels -> none derived from config.
    assert new_pos.stop_loss is None
    assert new_pos.take_profit is None


def test_zero_pct_disables_sl_tp_legs():
    """A 0 pct disables that leg (no SL/TP), never pinning to entry price.

    Guards the shannon's-demon regression: stop_loss=0 + take_profit=0 used to
    set SL = entry_price, so every rebalance stopped out the next bar.
    """
    config = RiskConfig(stop_loss_pct=0.0, take_profit_pct=0.0, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
    )
    # Even a bar gapping far below entry must NOT trigger a stop when SL is 0.
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=92.0,
        high=93.0,
        low=85.0,
        close=86.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, config)
    assert event is None
    assert new_pos.stop_loss is None
    assert new_pos.take_profit is None


def test_no_level_stays_none_even_when_trailing_enabled():
    """A position with no strategy-set levels stays level-less; trailing cannot
    derive a level from config (config-level SL/TP was removed)."""
    config = RiskConfig(stop_loss_pct=0.0, take_profit_pct=0.1, trailing_stop=True)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=110.0,
        high=115.0,
        low=109.0,
        close=114.0,
        volume=1000,
    )
    new_pos, _ = check_position_risk(position, tick, config)
    assert new_pos.stop_loss is None
    assert new_pos.take_profit is None


def test_no_sl_tp_without_explicit_levels_short():
    """No config-level SL/TP fallback for shorts either."""
    config = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.15, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=-10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.short,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, config)
    assert event is None
    assert new_pos.stop_loss is None
    assert new_pos.take_profit is None


def test_auto_sl_trailed():
    """An existing SL trails up with price when trailing_stop is enabled."""
    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=True)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,  # already set by risk module on prior tick
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    new_pos, event = check_position_risk(position, tick, config)
    # SL moved up: 115 * (1 - 0.1) = 103.5 > 90.0
    assert new_pos.stop_loss == 103.5
    assert event is None


# ── update_trailing_stop (legacy) ──────────────────────────────────


def test_trailing_sl_moves_up_long():
    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=True)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    new_pos = update_trailing_stop(position, tick, config)
    assert new_pos.stop_loss
    assert position.stop_loss
    assert new_pos.stop_loss > position.stop_loss


def test_trailing_sl_disabled_when_not_configured():
    """update_trailing_stop leaves the stop unchanged when trailing is off."""
    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.2, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    new_pos = update_trailing_stop(position, tick, config)
    assert new_pos is position  # unchanged


# ── check_risk (returns (events, portfolio)) ──────────────────────────


def test_no_risk_on_different_symbol(risk_config):
    from src.bt.state import PortfolioState, EquityPoint

    position = Position(
        symbol="GOOGL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=90.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=10000,
        positions={"GOOGL": (position,)},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=get_ts("2025-01-01"),
                equity=10000,
                cash=10000,
                positions_value=0.0,
            ),
        ),
        initial_capital=10000,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=80.0,
        high=85.0,
        low=75.0,
        close=78.0,
        volume=1000,
    )
    events, updated_pf = check_risk(portfolio, tick, risk_config)
    assert events == ()
    # Portfolio unchanged when no matching symbol
    assert updated_pf is portfolio


def test_check_risk_persists_strategy_sl_tp_levels():
    """Strategy-set SL/TP levels on the Position persist through check_risk."""
    from src.bt.state import PortfolioState, EquityPoint

    config = RiskConfig(stop_loss_pct=0.0, take_profit_pct=0.0, trailing_stop=False)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=95.0,
        take_profit=115.0,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=10000,
        positions={"AAPL": (position,)},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=get_ts("2025-01-01"),
                equity=10000,
                cash=10000,
                positions_value=0.0,
            ),
        ),
        initial_capital=10000,
    )
    tick = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1000,
    )
    events, new_pf = check_risk(portfolio, tick, config)
    assert events == ()
    updated_pos = new_pf.positions["AAPL"][0]
    # Strategy-set levels are preserved unchanged (only source of truth).
    assert updated_pos.stop_loss == pytest.approx(95.0)
    assert updated_pos.take_profit == pytest.approx(115.0)


def test_check_risk_trails_auto_sl():
    """Trailing stop is persisted across calls via the returned portfolio."""
    from src.bt.state import PortfolioState, EquityPoint

    config = RiskConfig(stop_loss_pct=0.1, take_profit_pct=0.5, trailing_stop=True)
    position = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=get_ts("2025-01-01"),
        stop_loss=None,
        take_profit=None,
        last_price=100.0,
        type=ActionType.long,
    )
    portfolio = PortfolioState(
        cash=10000,
        positions={"AAPL": (position,)},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=get_ts("2025-01-01"),
                equity=10000,
                cash=10000,
                positions_value=0.0,
            ),
        ),
        initial_capital=10000,
    )

    # First tick: SL set to 90 (entry * 0.9), then trailed to 103.5
    tick1 = Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="AAPL",
        open=105.0,
        high=115.0,
        low=104.0,
        close=110.0,
        volume=1000,
    )
    events, pf = check_risk(portfolio, tick1, config)
    assert events == ()
    pos1 = pf.positions["AAPL"][0]
    assert pos1.stop_loss == pytest.approx(103.5)  # 115 * 0.9

    # Second tick: SL trails further up (TP=150.0, well above, no trigger)
    tick2 = Candle(
        timestamp=get_ts("2025-01-03"),
        symbol="AAPL",
        open=110.0,
        high=120.0,
        low=109.0,
        close=118.0,
        volume=1000,
    )
    events, pf2 = check_risk(pf, tick2, config)
    assert events == ()
    pos2 = pf2.positions["AAPL"][0]
    assert pos2.stop_loss == pytest.approx(108.0)  # 120 * 0.9
