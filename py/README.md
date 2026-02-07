# IBKR PY

CLI package for testing strategies and visualizing signals.

# Development

The package leverages `ux` for package management and tool running.

# Running the project

The project uses `uv` as a package manager.
For running the project use `uv run main.py ...`

# Testing the project

Running the tests suite is done by running `uv run pytest`
Running the suite in watch mode is done like so: `find . -type f -name "*.py" | entr uv run pytest`

# Type checking & formatting

The project relies on `ty` for LSP functionality and `ruff` for formatting.
To typecheck: `uv tool run ty check`
To format: `uv tool run ruff format`
