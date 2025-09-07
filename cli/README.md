# Interactive Brokers CLI

These are scripts to interact with the IBKR REST API.
Each script should be mostly isolated, with the exception of common utilities.
Use `julia` as the programming language
Prefer a modular architecture, keeping files small.
Use simple code, following best practices and design patterns for more complex cases.
The code should be production ready.

## Usage

The whole package is accessible via a cli interface.
It works with the command + subcommand pattern (similar to git or the `click` python package).

### Auth

Running `ibcli auth status` will make the appropriate call and return auth status

### Candles

It's possible to download candles using the `candles` subcommand.
This will download the appropriate OHLCV candles.

Examples:
- `ibcli auth status`
- `ibcli candles AAPL -i 1d -p 90d -t STK`

