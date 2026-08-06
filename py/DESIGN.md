# IBKR Library — Agent-Friendly CLI

> **Historical design note.** This file describes an earlier architecture (a `py`
> binary with `kalman`/`hmm`/`spread`/`mx`/`screen`/`ind` modules and a pipe-based
> JSON-lines CLI). That layout is superseded. The current CLI is `ibkr` with two
> groups, `bt` and `data` — see **README.md** for the live quickstart and command
> reference.

## Status: SUPERSEDED

All 8 modules have CLI groups. Piped JSON lines composition works.

## Quick Start

```bash
# Install
cd py && uv pip install -e .

# Query data
py data query AAPL --from 2026-01-01 --to 2026-01-15

# Download + query
py data dl AAPL MSFT --from 2026-01-01

# Pipe: data → kalman
py data query AAPL --from 2026-01-01 | py kalman run --stdin

# Pipe: data → indicator
py data query AAPL --from 2026-01-01 | py ind rsi --window 14

# Pipe: data → kalman → indicator (future: kalman output format needs indicator compat)
py data query AAPL --from 2026-01-01 | py kalman run --stdin

# Pairs analysis
py spread analyze AAPL MSFT --from 2024-01-01

# HMM regime detection
py hmm fit AAPL --from 2024-01-01 --n-regimes 3

# Backtest
py bt run strategy.json

# IS/OOS walk-forward validation
py bt split strategy.json --folds 4
py bt split strategy.json --is-end 2020-12-31

# Walk-forward param optimization (IS-tune → OOS-validate per fold)
py bt optimize strategy.json '{"strategy_params":{"ma_slow":[50,100,200]}}' --folds 4

# Screen
py screen run breakout_screen universe.json --param fast=50
```

## Architecture

```
py/
├── main.py                  # Click root group — delegates to sub-groups
├── pyproject.toml           # [project.scripts] "py = main:main"
├── src/
│   ├── shared/              # NEW: DB helpers (extracted from utils.py)
│   │   ├── __init__.py
│   │   └── db.py            # query_candles — pure sqlite3, no ORM
│   ├── data/                # NEW: market data module
│   │   ├── __init__.py
│   │   └── cli.py           # py data {dl,query,preview}
│   ├── indicators/          # NEW: technical indicators with CLI
│   │   ├── __init__.py
│   │   └── cli.py           # py ind {ema,rsi,atr,macd,bbands,adx,vol}
│   ├── kalman/
│   │   ├── cli.py           # py kalman {run,pairs} + legacy kalman() fn
│   │   ├── pure.py          # unchanged
│   │   └── types.py         # unchanged
│   ├── hmm/
│   │   ├── cli.py           # py hmm {fit,predict}
│   │   └── ...              # unchanged
│   ├── spread/
│   │   ├── cli.py           # py spread {analyze}
│   │   └── __init__.py      # (old spread() fn still in old code)
│   ├── mx/
│   │   ├── cli.py           # py mx {matrix}
│   │   └── __init__.py
│   ├── bt/
│   │   ├── cli.py           # py bt {run,analyze}
│   │   ├── __init__.py      # preserved: load_strategy, backtest, etc.
│   │   └── ...              # engine unchanged
│   ├── screen/
│   │   ├── cli.py           # py screen {run}
│   │   └── __init__.py      # preserved: run_screen, import_screen, etc.
│   └── utils.py             # old get_local_candles (still used by some tests)
```

## Pipe Format (JSON Lines)

Every command reads/writes JSON lines to stdout/stdin:

```jsonl
{
  "t": "2026-01-05T14:00:00Z",
  "o": 100.5,
  "h": 101.2,
  "l": 100.1,
  "c": 100.9,
  "v": 12345
}
```

- **data** outputs: `{t, o, h, l, c, v}`
- **kalman** outputs: `{t, filtered, predicted, upper_ci, lower_ci, residual, kalman_gain, velocity}`
- **kalman pairs** outputs: `{t, alpha, beta, spread, t_stat, innovation_S}`
- **hmm** outputs: `{t, regime, regime_0, regime_1, ...}`
- **indicators** output: `{t, ema_20}` or `{t, rsi}` or `{t, upper, middle, lower}` etc.
- **mx** outputs: JSON object `{symbols, correlation, cointegration_pvalues}`
- **spread** outputs: `{t, t_stat, beta, alpha, spread, innovation_S}`
- **bt** outputs: text summary or JSON metrics+trades
- **screen** outputs: text table or JSON results

## Stdin/Stderr Convention

- **stdout**: machine-readable JSON lines
- **stderr**: human-readable progress, stats, errors
- `--stdin` flag: read OHLCV from stdin instead of DB query
- Without `--stdin`: `--symbol` + `--from/--to` query the local DB

## Unchanged

- `bt/engine/` — functional backtesting core (pure, well-tested)
- `bt/indicators.py` — still exists (imported by old strategies), duplicated in `src/indicators/cli.py`
- `syncm/` — IBKR sync layer (async, with API client)
- `nd/`, `pnd/` — deviation plots (no CLI, less agent-relevant)
- `screen/__init__.py` — full screen engine preserved

## Un-ported (from old main.py)

These old root commands are superseded by subgroups:

| Old command                 | New equivalent                         |
| --------------------------- | -------------------------------------- |
| `python main.py mx ...`     | `py mx matrix ...`                     |
| `python main.py spread ...` | `py spread analyze ...`                |
| `python main.py nd ...`     | Not ported (visualization)             |
| `python main.py pnd ...`    | Not ported (visualization)             |
| `python main.py hmm ...`    | `py hmm fit/predict ...`               |
| `python main.py kalman ...` | `py kalman run ...`                    |
| `python main.py bt ...`     | `py bt run ...`                        |
| `python main.py sync ...`   | `py data dl ...`                       |
| `python main.py signal ...` | Not ported (live trading out of scope) |
| `python main.py screen ...` | `py screen run ...`                    |
