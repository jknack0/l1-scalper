"""Run the live paper trading bot.

Connects to Databento live feed + Tradovate demo, runs all regime-gated
models with trailing stop exits.

Usage:
    python scripts/run_paper.py
    python scripts/run_paper.py --symbol MESM6
    python scripts/run_paper.py --dry-run  # data feed only, no orders
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from src.live.bot import BotConfig, LiveBot
from src.live.tradovate_executor import TradovateExecutor


@click.command()
@click.option("--symbol", default="MESM6", help="MES contract symbol (e.g., MESM6 for June 2026).")
@click.option("--dry-run", is_flag=True, help="Connect to data feed but don't place orders.")
@click.option("--verbose", "-v", is_flag=True)
def main(symbol: str, dry_run: bool, verbose: bool) -> None:
    """Run the paper trading bot."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = BotConfig.default_paper()
    config.symbol = symbol

    if dry_run:
        click.echo("DRY RUN: will connect to data feed but not place orders.")
        # Use a dummy executor that logs but doesn't send
        executor = DryRunExecutor()
    else:
        executor = TradovateExecutor()

    bot = LiveBot(config=config, executor=executor)

    click.echo(f"Starting paper trading bot...")
    click.echo(f"  Symbol: {symbol}")
    click.echo(f"  Models: {[mc.regime_name for mc in config.models]}")
    click.echo(f"  Daily loss limit: ${config.daily_loss_limit}")
    click.echo()

    asyncio.run(bot.run())


class DryRunExecutor:
    """Dummy executor that logs orders but doesn't send them."""

    def authenticate(self) -> None:
        logging.getLogger(__name__).info("DRY RUN: authenticate()")

    def find_contract(self, symbol: str) -> int:
        logging.getLogger(__name__).info("DRY RUN: find_contract(%s) -> 99999", symbol)
        return 99999

    def place_market_order(self, contract_id: int, action: str, qty: int = 1) -> dict:
        logging.getLogger(__name__).info("DRY RUN: market %s x%d", action, qty)
        return {"orderId": 0}

    def place_stop_order(self, contract_id: int, action: str, stop_price: float,
                         qty: int = 1) -> dict:
        logging.getLogger(__name__).info("DRY RUN: stop %s x%d @ %.2f", action, qty, stop_price)
        return {"orderId": 0}

    def cancel_order(self, order_id: int) -> dict:
        logging.getLogger(__name__).info("DRY RUN: cancel order %d", order_id)
        return {}

    def get_positions(self) -> list:
        return []

    def get_orders(self) -> list:
        return []

    def flatten_position(self, contract_id: int) -> None:
        logging.getLogger(__name__).info("DRY RUN: flatten %d", contract_id)


if __name__ == "__main__":
    main()
