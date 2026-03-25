"""Precompute macro HMM v2 features from 1-second OHLCV bars.

Processes one year at a time, saves feature vectors to parquet:
    data/features/macro_hmm_v2/year=YYYY/data.parquet

Uses 5-minute windows (300 × 1-sec bars) with 7 features per window.

Usage:
    python scripts/precompute_macro_features_v2.py
    python scripts/precompute_macro_features_v2.py --years 2020-2026
    python scripts/precompute_macro_features_v2.py --years 2025 --window 300
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

from src.regime.macro_features_v2 import (
    DEFAULT_MACRO_WINDOW,
    MACRO_FEATURE_NAMES,
    macro_features_from_1s_bars,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OHLCV_1S_DIR = DATA_DIR / "parquet"
FEATURES_DIR = DATA_DIR / "features" / "macro_hmm_v2"


def _available_years() -> list[int]:
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
@click.option("--window", default=DEFAULT_MACRO_WINDOW, type=int,
              help=f"Number of 1-sec bars per window. Default: {DEFAULT_MACRO_WINDOW} (5 min).")
@click.option("--force", is_flag=True, help="Recompute even if features already exist.")
def main(years: str | None, window: int, force: bool) -> None:
    """Precompute macro HMM v2 features from 1-second OHLCV bars."""
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
    click.echo(f"Window: {window} bars ({window}s = {window // 60} min)")
    click.echo(f"Features: {len(MACRO_FEATURE_NAMES)} ({', '.join(MACRO_FEATURE_NAMES)})")
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
        click.echo(f"  {year}: loading (close, volume, timestamp only)...")
        needed_cols = ["close", "volume", "timestamp"]
        # Use ParquetFile to avoid Hive partition schema merge issues
        pf = pq.ParquetFile(in_path)
        available = pf.schema_arrow.names
        cols_to_read = [c for c in needed_cols if c in available]
        df = pf.read(columns=cols_to_read).to_pandas()
        n_bars = len(df)

        click.echo(f"  {year}: computing macro features from {n_bars:,} 1-sec bars...")
        feats, timestamps = macro_features_from_1s_bars(df, window=window)
        del df

        if len(feats) == 0:
            click.echo(f"  [WARN] {year}: no valid features")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        columns = {
            name: feats[:, i].astype(np.float64)
            for i, name in enumerate(MACRO_FEATURE_NAMES)
        }
        columns["timestamp"] = timestamps
        table = pa.table(columns)
        pq.write_table(table, out_path, compression="zstd")

        elapsed = time.time() - t0
        click.echo(f"  {year}: {len(feats):,} feature vectors in {elapsed:.1f}s")
        total_rows += len(feats)

    elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {total_rows:,} total vectors in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
