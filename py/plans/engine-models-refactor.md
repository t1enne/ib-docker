# Engine + Models Refactor Plan

## Scope

- Core engine: merge walk-forward and backtest engines into a single engine entrypoint.
- Models: refactor StrategyModel and related helpers to align with style defaults.
- Config handling: pass full strategy config object instead of drilling kwargs.

## Defaults

- Configs as dataclasses, passed as whole objects.
- Prefer small pure helpers and assertions for preconditions.
- Keep mutation only where state is inherently mutable (portfolio, risk, market data).
- Keep functions < 50 LOC and classes < 150 LOC (split helpers as needed).

## Steps

1. Introduce config dataclasses (engine + models)
   - Add config types to engine module (dates, symbols, strategy params, portfolio, execution, HMM).
   - Make the existing Strategy dataclass the top-level config, used by engine directly.

2. Merge engines
   - Remove WalkForwardEngine class.
   - Expand BacktestEngine to accept Strategy (or BacktestConfig) and run the same workflow.
   - Keep walk-forward window computation inside BacktestEngine for future extension.

3. Refactor engine flow
   - Extract data loading, tick grouping, preseed, and trading loop into small helpers.
   - Normalize tick grouping for both preseed and live loop.
   - Keep outputs identical (results, z-score df, regime df).

4. Refactor StrategyModel
   - Encapsulate HMM state and z-score buffers in dataclass state objects.
   - Split HMM update into small helpers with assertions for preconditions.
   - Keep StrategyModel public surface unchanged (z_score, current_regime, market_data).

5. Update public API and tests
   - Update src/bt/__init__.py backtest() to instantiate BacktestEngine only.
   - Update any tests and call sites to new engine signature.

## Verification

- Run: `uv run pytest`
- Optional: `uv tool run ty check src/bt/`
