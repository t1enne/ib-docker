MAKEFLAGS += --no-print-directory --silent

SRC_DIRS := src/bt src/kalman src/syncm src/data src/db src/indicators src/market_data src/mx src/screen
SRC      := $(SRC_DIRS) 
.PHONY: help lint format check test run exec

help:
	@echo "Usage:"
	@echo "  make run  <cli-args>  — Run the CLI"
	@echo "  make check            — Lint + format + typecheck + tests"
	@echo "  make test             — Run all tests"
	@echo "  make lint             — Ruff check"
	@echo "  make format           — Ruff format"
	@echo "  make typecheck        — Ty check"
	@echo ""
	@echo "CLI commands via make run/exec:"
	@echo "  make run bt run <yaml>           — Run backtest"
	@echo "  make run bt run <yaml> --format jsonl  — JSON output"
	@echo "  make run bt analyze <yaml>       — Analyze strategy"
	@echo "  make run data query <sym>        — Query candles"
	@echo "  make run data dl <sym>...        — Download candles"
	@echo "  make run bt --help               — Click-style help"
	@echo "  make run data --help             — Data subcommand help"

lint:
	uv run ruff check $(SRC)

format:
	uv run ruff format $(SRC)

typecheck:
	uv run ty check $(SRC)

test:
	uv run pytest $(SRC_DIRS) tests -v

test-fast:
	uv run pytest $(SRC_DIRS) tests --no-header -q

check: lint format typecheck test-fast

run:
	@if [ -n "$(ARGS)" ]; then \
		uv run main.py $(ARGS); \
	else \
		uv run main.py $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)); \
	fi

exec: run

# Swallow extra words passed after make run so Make doesn't error.
# e.g. make run bt run strat.yaml → args consumed by 'run'; 'bt'/'run'/'strat.yaml' matched here.
%::
	@true
