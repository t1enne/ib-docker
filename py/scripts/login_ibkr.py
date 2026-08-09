"""Automate IBKR Gateway login via Playwright.

Usage:
    export IBKR_USERNAME=...
    export IBKR_PASSWORD=...
    export TRADING_MODE=paper   # or "live"

    uv run python scripts/login_ibkr.py

Or pass credentials as CLI args:
    uv run python scripts/login_ibkr.py --username U123456 --password s3cret --mode paper
"""

import sys
from pathlib import Path

# When running as uv run python scripts/login_ibkr.py, sys.path[0] is set to
# scripts/ rather than the project root. Insert the project root so that
# imports from src/ resolve correctly.
_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

import asyncio
import os
from typing import Literal

from playwright.async_api import async_playwright

TradingMode = Literal["paper", "live"]


def load_env(path: str | Path = ".env") -> None:
    """Load a .env file into os.environ. Never overrides existing env vars."""

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = val


async def login_ibkr(username: str, password: str, mode: TradingMode = "paper") -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        try:
            await page.goto("https://localhost:5000")

            # Enable paper trading mode
            if mode == "paper":
                await page.click('label[for="toggle1"]')

            await page.wait_for_timeout(500)

            await page.fill(
                'input[name="username"], #username, input[type="text"]',
                username,
            )
            await page.fill(
                'input[name="password"], #password, input[type="password"]',
                password,
            )
            await page.click(
                'button[type="submit"], input[type="submit"], .login-button',
            )

            # Wait for the success text to appear
            await page.get_by_text("Client login succeeds").wait_for()
            await page.wait_for_timeout(1000)

            print("Login attempt completed")
            # Browser stays open so you can interact with the Gateway
        except Exception as exc:
            print(f"Login failed: {exc}", file=sys.stderr)
            await browser.close()
            sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Login to IBKR Gateway")
    parser.add_argument("--username", default=os.environ.get("IBKR_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("IBKR_PASSWORD"))
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=os.environ.get("TRADING_MODE", "paper"),
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        print(
            "Provide IBKR_USERNAME and IBKR_PASSWORD via env vars, "
            "or pass --username / --password on the command line.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(login_ibkr(args.username, args.password, args.mode))


if __name__ == "__main__":
    load_env()
    main()
