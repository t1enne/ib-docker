MAKEFLAGS += --no-print-directory --silent

SRC_DIRS := src/bt src/kalman src/syncm src/data src/db src/indicators src/market_data src/mx src/screen
SRC      := $(SRC_DIRS) 
TEST_DIRS := $(SRC_DIRS) tests

.PHONY: lint format check test

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
	uv run main.py $(filter-out run,$(MAKECMDGOALS))
