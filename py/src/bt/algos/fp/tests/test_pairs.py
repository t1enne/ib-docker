from src.bt.algos.base_pairs_strategy_fp import init_state
from src.bt.algos.pairs_trading_fp import (
    calculate_zscore,
    generate_signals,
    process_tick,
)
from src.bt.types import Tick, ActionType
from src.utils import get_ts


def test_init_pairs_state():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=100,
    )
    assert state.symbols == ["AAPL", "MSFT"]
    assert state.config.entry_threshold == 2.0
    assert state.config.rolling_window_size == 100


def test_calculate_zscore_insufficient_data():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=100,
    )
    assert calculate_zscore(state) is None


def test_calculate_zscore_with_data():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=10,
    )
    # Add some data
    for i in range(10):
        ts = get_ts(f"2023-01-{i + 1}")
        print(ts)
        state.historical_data["AAPL"].loc[ts, "close"] = 100 + i
        state.historical_data["MSFT"].loc[ts, "close"] = 200 + i
    print(state.historical_data["MSFT"])
    z = calculate_zscore(state)
    assert z is not None
    assert isinstance(z, float)


def test_generate_signals_no_position_entry():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=100,
    )
    ts = get_ts("2023-01-01")
    state.z_scores[ts] = 2.5
    state.pending_ticks[ts] = {"AAPL": 100.0, "MSFT": 200.0}
    new_state, signals = generate_signals(state, ts)
    assert len(signals) == 2
    assert signals[0].action == ActionType.long
    assert signals[0].symbol == "MSFT"  # Long second symbol
    assert new_state.positions["AAPL"] == -1.0  # Short first
    assert new_state.positions["MSFT"] == 1.0  # Long second


def test_generate_signals_close_position():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=100,
    )
    ts = get_ts("2023-01-01")
    state.positions = {"AAPL": 1.0, "MSFT": -1.0}
    state.z_scores[ts] = 0.3
    state.pending_ticks[ts] = {"AAPL": 100.0, "MSFT": 200.0}
    new_state, signals = generate_signals(state, ts)
    assert len(signals) == 2
    assert signals[0].action == ActionType.close
    assert new_state.positions["AAPL"] == 0.0


def test_process_tick():
    state = init_state(
        ["AAPL", "MSFT"],
        entry_z=2.0,
        stop_loss=0.5,
        take_profit=3.0,
        rolling_window_size=10,
    )
    # Add data for both
    ts = get_ts("2023-01-01")
    tick1 = Tick(
        symbol="AAPL",
        timestamp=ts,
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        volume=1000,
    )
    tick2 = Tick(
        symbol="MSFT",
        timestamp=ts,
        open=199.0,
        high=201.0,
        low=198.0,
        close=200.0,
        volume=2000,
    )
    state, signals1 = process_tick(state, tick1)
    state, signals2 = process_tick(state, tick2)
    # Since insufficient data for z-score, pending ticks remain
    assert len(state.pending_ticks[ts]) == 2

