"""Download 1-minute OHLCV bars from Databento for MES.

Saves to data/ohlcv_1m/year=YYYY/data.parquet, one file per year.

Usage:
    python scripts/download_ohlcv.py                    # 2020-present
    python scripts/download_ohlcv.py --years 2023-2025
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import databento as db
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import DatabentoSettings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA_DIR / "ohlcv_1m"


@click.command()
@click.option("--years", default="2020-2026",
              help="Year range (e.g., '2020-2026') or comma-separated.")
@click.option("--force", is_flag=True, help="Re-download even if file exists.")
def main(years: str, force: bool) -> None:
    """Download 1-minute OHLCV bars from Databento."""
    settings = DatabentoSettings()
    client = db.Historical(key=settings.api_key)

    if "-" in years:
        start, end = years.split("-")
        year_list = list(range(int(start), int(end) + 1))
    else:
        year_list = [int(y) for y in years.split(",")]

    click.echo(f"Dataset: {settings.dataset}")
    click.echo(f"Symbol: {settings.symbol}")
    click.echo(f"Years: {year_list}")
    click.echo(f"Output: {OUT_DIR}/")
    click.echo()

    for year in year_list:
        out_dir = OUT_DIR / f"year={year}"
        out_path = out_dir / "data.parquet"

        if out_path.exists() and not force:
            t = pq.read_table(out_path)
            click.echo(f"  [EXISTS] {year}: {t.num_rows:,} rows (use --force to re-download)")
            continue

        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        click.echo(f"  Downloading {year} ({start_date} to {end_date})...")
        t0 = time.time()

        try:
            data = client.timeseries.get_range(
                dataset=settings.dataset,
                schema="ohlcv-1m",
                symbols="MES.c.0",
                stype_in="continuous",
                start=start_date,
                end=end_date,
            )

            df = data.to_df()
            elapsed = time.time() - t0
            click.echo(f"    {len(df):,} rows downloaded in {elapsed:.1f}s")

            if len(df) == 0:
                click.echo(f"    [WARN] No data for {year}, skipping")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, engine="pyarrow", compression="zstd")
            click.echo(f"    Saved {out_path}")

        except Exception as e:
            click.echo(f"    [ERROR] {year}: {e}")
            continue

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
