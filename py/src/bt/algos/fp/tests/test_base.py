import pandas as pd
import src.bt.algos.base_pairs_strategy_fp as bps
from src.bt.types import ActionType
from src.utils import get_ts


def test_init_state():
    state = bps.init_state(
        ["AAPL", "MSFT"],
        entry_threshold=2.0,
        exit_threshold=0.5,
        rolling_window_size=100,
    )
    assert state.symbols == ["AAPL", "MSFT"]
    assert state.config.entry_threshold == 2.0
    assert len(state.positions) == 2
    assert all(pos == 0.0 for pos in state.positions.values())


def test_add_historical_data():
    state = bps.init_state(
        ["AAPL"], entry_threshold=2.0, exit_threshold=0.5, rolling_window_size=100
    )
    data = {
        "AAPL": pd.DataFrame(
            {"Close": [100, 101]},
            index=pd.date_range("2023-01-01", periods=2),
        )
    }
    new_state = bps.add_historical_data(state, data)
    assert len(new_state.historical_data["AAPL"]) == 2
    assert new_state.historical_data["AAPL"].iloc[0]["close"] == 100


def test_create_signal():
    state = bps.init_state(
        ["AAPL"], entry_threshold=2.0, exit_threshold=0.5, rolling_window_size=100
    )
    ts = get_ts("2023-01-01")
    state.z_scores[ts] = 1.5
    state.pending_ticks[ts] = {"AAPL": 100.0}
    signal = bps.create_signal(ActionType.long, "AAPL", ts, state)
    assert signal.action == ActionType.long
    assert signal.symbol == "AAPL"
    assert signal.z_score == 1.5
    assert signal.price == 100.0
