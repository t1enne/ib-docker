# Interactive Brokers Scripts

These are scripts to interact with the IBKR REST API.
Each script should be mostly isolated, with the exception of common utilities.
Use `julia` as the programming langauge
Prefer a modular architecture, keeping files small.
Use simple code, following best practices and design patterns for more complex cases.
The code should be production ready.

# Architecture

The whole package should be accessible via a cli interface.
It should work with the command + subcommand pattern (similar to git or the `click` python package).

# Auth

Running `ibcli auth status` will make the approriate call and return auth status

# Candles

It's possible to download candles using the `candles` subcommand.
This will download the appropriate OHLCV candles and store them in a sqlite db.

Ex:
- `ibcli candles AAPL -i 1d -p 90d -t STK`

