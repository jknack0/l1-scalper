"""Catch up L1 and OHLCV data from Databento to present.

Detects the last date in existing data and downloads everything from
there to yesterday. Appends to the existing yearly parquet files.

Usage:
    python scripts/catchup_data.py                  # both L1 and OHLCV
    python scripts/catchup_data.py --l1-only        # just L1 ticks
    python scripts/catchup_data.py --ohlcv-only     # just 1-sec bars
    python scripts/catchup_data.py --dry-run        # show what would be downloaded
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import click
import databento as db
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import DatabentoSettings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
L1_DIR = DATA_DIR / "l1"
OHLCV_DIR = DATA_DIR / "parquet"

logger = logging.getLogger(__name__)


def _find_last_date(data_dir: Path, col: str = "timestamp") -> date | None:
    """Find the last date in yearly parquet files."""
    latest = None
    for p in sorted(data_dir.iterdir()):
        if not p.name.startswith("year="):
            continue
        parquet_path = p / "data.parquet"
        if not parquet_path.exists():
            continue
        try:
            # Use ParquetFile to avoid Hive partition schema merge issues
            pf = pq.ParquetFile(parquet_path)
            t = pf.read(columns=[col])
            ts = t.column(col).to_numpy()
            max_ts = np.datetime_as_string(ts.max(), unit="D")
            d = date.fromisoformat(max_ts)
            if latest is None or d > latest:
                latest = d
        except Exception as e:
            click.echo(f"  [WARN] Could not read {parquet_path}: {e}")
    return latest


def _catchup_l1(client: db.Historical, settings: DatabentoSettings,
                start: date, end: date) -> None:
    """Download L1 (TBBO) data and append to yearly files."""
    click.echo(f"\n  Downloading L1 TBBO: {start} to {end}")

    # Group by year
    current = start
    year_batches: dict[int, list[pd.DataFrame]] = {}

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        next_day = current + timedelta(days=1)
        click.echo(f"    {current}...", nl=False)
        t0 = time.time()

        try:
            data = client.timeseries.get_range(
                dataset=settings.dataset,
                symbols=["MES.c.0"],
                schema="tbbo",
                stype_in="continuous",
                start=str(current),
                end=str(next_day),
            )
            df = data.to_df()
            elapsed = time.time() - t0

            if len(df) == 0:
                click.echo(f" no data ({elapsed:.1f}s)")
                current = next_day
                continue

            click.echo(f" {len(df):,} rows ({elapsed:.1f}s)")

            year = current.year
            if year not in year_batches:
                year_batches[year] = []
            year_batches[year].append(df)

        except Exception as e:
            click.echo(f" ERROR: {e}")

        current = next_day

    # Append to yearly files
    for year, dfs in year_batches.items():
        if not dfs:
            continue

        new_data = pd.concat(dfs, ignore_index=True)
        out_dir = L1_DIR / f"year={year}"
        out_path = out_dir / "data.parquet"

        if out_path.exists():
            click.echo(f"\n  Appending {len(new_data):,} L1 rows to year={year}...")
            existing = pq.ParquetFile(out_path).read().to_pandas()
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            del existing
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            combined = new_data.sort_values("timestamp")

        combined.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        click.echo(f"  Saved {len(combined):,} total L1 rows to {out_path}")
        del combined, new_data


def _catchup_ohlcv(client: db.Historical, settings: DatabentoSettings,
                   start: date, end: date) -> None:
    """Download 1-sec OHLCV data and append to yearly files."""
    click.echo(f"\n  Downloading OHLCV 1-sec: {start} to {end}")

    current = start
    year_batches: dict[int, list[pd.DataFrame]] = {}

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        next_day = current + timedelta(days=1)
        click.echo(f"    {current}...", nl=False)
        t0 = time.time()

        try:
            data = client.timeseries.get_range(
                dataset=settings.dataset,
                symbols=["MES.c.0"],
                schema="ohlcv-1s",
                stype_in="continuous",
                start=str(current),
                end=str(next_day),
            )
            df = data.to_df()
            elapsed = time.time() - t0

            if len(df) == 0:
                click.echo(f" no data ({elapsed:.1f}s)")
                current = next_day
                continue

            click.echo(f" {len(df):,} rows ({elapsed:.1f}s)")

            year = current.year
            if year not in year_batches:
                year_batches[year] = []
            year_batches[year].append(df)

        except Exception as e:
            click.echo(f" ERROR: {e}")

        current = next_day

    # Append to yearly files
    for year, dfs in year_batches.items():
        if not dfs:
            continue

        new_data = pd.concat(dfs, ignore_index=True)
        out_dir = OHLCV_DIR / f"year={year}"
        out_path = out_dir / "data.parquet"

        if out_path.exists():
            click.echo(f"\n  Appending {len(new_data):,} OHLCV rows to year={year}...")
            existing = pq.ParquetFile(out_path).read().to_pandas()
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            del existing
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            combined = new_data.sort_values("timestamp")

        combined.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        click.echo(f"  Saved {len(combined):,} total OHLCV rows to {out_path}")
        del combined, new_data


@click.command()
@click.option("--l1-only", is_flag=True, help="Only download L1 ticks.")
@click.option("--ohlcv-only", is_flag=True, help="Only download OHLCV bars.")
@click.option("--dry-run", is_flag=True, help="Show date ranges without downloading.")
@click.option("--verbose", "-v", is_flag=True)
def main(l1_only: bool, ohlcv_only: bool, dry_run: bool, verbose: bool) -> None:
    """Catch up L1 and OHLCV data from Databento to present."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    yesterday = date.today() - timedelta(days=1)

    do_l1 = not ohlcv_only
    do_ohlcv = not l1_only

    click.echo("=" * 60)
    click.echo("DATA CATCHUP")
    click.echo("=" * 60)

    if do_l1:
        l1_last = _find_last_date(L1_DIR)
        l1_start = l1_last + timedelta(days=1) if l1_last else date(2025, 1, 1)
        click.echo(f"  L1:    last={l1_last}, need {l1_start} to {yesterday} ({(yesterday - l1_start).days} days)")

    if do_ohlcv:
        ohlcv_last = _find_last_date(OHLCV_DIR)
        ohlcv_start = ohlcv_last + timedelta(days=1) if ohlcv_last else date(2025, 1, 1)
        click.echo(f"  OHLCV: last={ohlcv_last}, need {ohlcv_start} to {yesterday} ({(yesterday - ohlcv_start).days} days)")

    if dry_run:
        click.echo("\n  --dry-run: not downloading.")
        return

    settings = DatabentoSettings()
    client = db.Historical(key=settings.api_key)

    if do_l1 and l1_start <= yesterday:
        _catchup_l1(client, settings, l1_start, yesterday)
    elif do_l1:
        click.echo("\n  L1: already up to date!")

    if do_ohlcv and ohlcv_start <= yesterday:
        _catchup_ohlcv(client, settings, ohlcv_start, yesterday)
    elif do_ohlcv:
        click.echo("\n  OHLCV: already up to date!")

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
