from typing import List, Tuple
import asyncio
import pandas as pd
import src.bt.algos.base_pairs_strategy_fp as bps
from src.bt.types import Tick, TradeSignal, ActionType
from src.utils import calculate_zscore_spread


def calculate_zscore(state: bps.StrategyState) -> float | None:
    # Use recent closes, limited to rolling window
    s1 = state.symbols[0]
    s1_closes = state.historical_data[s1]["close"].tail(
        state.config.rolling_window_size
    )
    print(s1_closes)
    s2 = state.symbols[1]
    s2_closes = state.historical_data[s2]["close"].tail(
        state.config.rolling_window_size
    )
    if len(s1_closes) < 2 or len(s2_closes) < 2:
        return None

    z_scores = calculate_zscore_spread(s1_closes, s2_closes)
    if z_scores.empty:
        return None
    return round(z_scores.iloc[-1], 2)


def generate_signals(
    state: bps.StrategyState, ts: pd.Timestamp
) -> Tuple[bps.StrategyState, List[TradeSignal]]:
    z_score = state.z_scores.get(ts, 0.0)
    signals = []
    new_positions = state.positions.copy()

    has_position = any(pos != 0 for pos in state.positions.values())
    if has_position and (
        abs(z_score) < state.config.exit_threshold
        or (state.config.take_profit and abs(z_score) > state.config.take_profit)
    ):
        # Close positions
        new_positions = {symbol: 0.0 for symbol in state.symbols}
        signals = [
            bps.create_signal(ActionType.close, state.symbols[0], ts, state),
            bps.create_signal(ActionType.close, state.symbols[1], ts, state),
        ]

    elif not has_position and abs(z_score) > state.config.entry_threshold:
        if z_score < -state.config.entry_threshold:
            new_positions[state.symbols[0]] = 1.0
            new_positions[state.symbols[1]] = -1.0
            signals = [
                bps.create_signal(ActionType.long, state.symbols[0], ts, state),
                bps.create_signal(ActionType.short, state.symbols[1], ts, state),
            ]
        elif z_score > state.config.entry_threshold:
            new_positions[state.symbols[0]] = -1.0
            new_positions[state.symbols[1]] = 1.0
            signals = [
                bps.create_signal(ActionType.long, state.symbols[1], ts, state),
                bps.create_signal(ActionType.short, state.symbols[0], ts, state),
            ]

    new_state = state.copy_with_updates(positions=new_positions)
    return new_state, signals


def process_tick(
    state: bps.StrategyState, tick: Tick
) -> Tuple[bps.StrategyState, List[TradeSignal]]:
    # Update historical_data in-place for performance
    state.historical_data[tick.symbol].loc[tick.timestamp, "close"] = tick.close
    # Update pending_ticks
    new_pending = state.pending_ticks.copy()
    if tick.timestamp not in new_pending:
        new_pending[tick.timestamp] = {}
    new_pending[tick.timestamp][tick.symbol] = tick.close

    new_state = state.copy_with_updates(pending_ticks=new_pending)

    # If we have data for both symbols at this timestamp, process
    if len(new_pending[tick.timestamp]) == len(state.symbols):
        z_score = calculate_zscore(new_state)
        if z_score is not None:
            new_z_scores = new_state.z_scores.copy()
            new_z_scores[tick.timestamp] = z_score
            new_state = new_state.copy_with_updates(z_scores=new_z_scores)
            new_state, signals = generate_signals(new_state, tick.timestamp)
            # Clear pending for this timestamp
            new_pending.pop(tick.timestamp, None)
            new_state = new_state.copy_with_updates(pending_ticks=new_pending)
            return new_state, signals

    return new_state, []


async def process_data_async(
    init_state: bps.StrategyState,
    ticks_queue: asyncio.Queue[Tick],
    order_queue: asyncio.Queue,
):
    state = init_state
    while True:
        tick = await ticks_queue.get()
        if tick is None:
            await order_queue.put(None)
            break
        state, signals = process_tick(state, tick)
        for signal in signals:
            await order_queue.put(signal)

