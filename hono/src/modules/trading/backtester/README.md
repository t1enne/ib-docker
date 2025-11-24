# Backtester

Here go the files for the backtesting/trading platform.
The lib should mimic zipline's event-based architecture for testing.
There's a main engine, and strategies that one can declare.
The engine will run the strategy, plot a graph of the ohlcv's provided and the equity curve, and display some stats about the run.

# Strategies

There's context being passed to the strategy to each tick, about some meta indicators like sentiment, market cycle
and other things.
Strategies should be able to allocate the position size dynamically, depending on the strenght of the setup.
A+ setups can allocate more compared to D setups.
The signals should not be just buy and sell, but should have a value between -1.0 and 1.0, to indicate the strength to the engine.
