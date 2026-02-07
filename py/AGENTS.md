# Development Guidelines

## Running Tests
```bash
uv run pytest src/bt/ -v
```

## Type Checking
```bash
uv tool run ty check src/bt/
```

## Project Structure
- `src/bt/algos/` - Trading strategies and models
- `src/bt/engine/` - Backtesting engines
- `src/bt/portfolio/` - Portfolio management
- `src/bt/execution/` - Execution handler for spreads/slippage

## Key Files
- `src/bt/algos/pairs_trading.py` - Pairs trading strategy
- `src/bt/algos/z_model.py` - Z-score model
- `src/bt/engine/backtest_engine.py` - Unified backtest engine
- `src/bt/engine/walk_forward_engine.py` - Walk-forward runner
