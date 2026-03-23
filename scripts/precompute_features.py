"""Precompute HMM features from 1-second OHLCV bars.

Processes one year at a time, saves feature vectors to parquet:
    data/features/hmm/year=YYYY/data.parquet

Uses 1-second bars from data/parquet/ for local regime detection
at scalping timescales (30s-3min trades).

Usage:
    python scripts/precompute_features.py                    # all available years
    python scripts/precompute_features.py --years 2020-2026
    python scripts/precompute_features.py --years 2025 --window 120
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime.features import FEATURE_NAMES, features_from_ohlcv_1m

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OHLCV_1S_DIR = DATA_DIR / "parquet"
FEATURES_DIR = DATA_DIR / "features" / "hmm"


def _available_years() -> list[int]:
    """Find all years with 1-second OHLCV data."""
    if not OHLCV_1S_DIR.exists():
        return []
    years = []
    for p in sorted(OHLCV_1S_DIR.iterdir()):
        if p.name.startswith("year=") and (p / "data.parquet").exists():
            years.append(int(p.name.split("=")[1]))
    return years


@click.command()
@click.option("--years", default=None,
              help="Year range (e.g., '2020-2026') or comma-separated. Default: all available.")
@click.option("--window", default=60, type=int,
              help="Number of 1-sec bars per feature window. Default: 60 (1 minute).")
@click.option("--force", is_flag=True, help="Recompute even if features already exist.")
def main(years: str | None, window: int, force: bool) -> None:
    """Precompute HMM features from 1-second OHLCV bars."""
    overall_t0 = time.time()

    if years is not None:
        if "-" in years:
            start, end = years.split("-")
            year_list = list(range(int(start), int(end) + 1))
        else:
            year_list = [int(y) for y in years.split(",")]
    else:
        year_list = _available_years()

    if not year_list:
        click.echo("No 1-second OHLCV data found in data/parquet/.")
        return

    click.echo(f"Years: {year_list}")
    click.echo(f"Window: {window} bars ({window}s)")
    click.echo(f"Input: {OHLCV_1S_DIR}/")
    click.echo(f"Output: {FEATURES_DIR}/")
    click.echo()

    total_rows = 0

    for year in year_list:
        in_path = OHLCV_1S_DIR / f"year={year}" / "data.parquet"
        out_dir = FEATURES_DIR / f"year={year}"
        out_path = out_dir / "data.parquet"

        if not in_path.exists():
            click.echo(f"  [SKIP] No OHLCV data for {year}")
            continue

        if out_path.exists() and not force:
            existing = pq.read_metadata(str(out_path)).num_rows
            click.echo(f"  [EXISTS] {year}: {existing:,} rows (use --force to recompute)")
            total_rows += existing
            continue

        t0 = time.time()
        click.echo(f"  {year}: loading...")
        df = pq.read_table(in_path).to_pandas()
        n_bars = len(df)

        click.echo(f"  {year}: computing features from {n_bars:,} 1-sec bars...")
        feats = features_from_ohlcv_1m(df, window=window)
        del df

        if len(feats) == 0:
            click.echo(f"  [WARN] {year}: no valid features")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            name: feats[:, i].astype(np.float64)
            for i, name in enumerate(FEATURE_NAMES)
        })
        pq.write_table(table, out_path, compression="zstd")

        elapsed = time.time() - t0
        click.echo(f"  {year}: {len(feats):,} feature vectors in {elapsed:.1f}s")
        total_rows += len(feats)

    elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {total_rows:,} total vectors in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
