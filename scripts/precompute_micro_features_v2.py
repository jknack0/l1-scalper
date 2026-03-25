"""Precompute micro HMM v2 features from raw L1 tick data.

Uses real bid/ask spread (not high-low proxy) and order flow imbalance.
Processes one year at a time, saves to parquet:
    data/features/micro_hmm_v2/year=YYYY/data.parquet

Strategy: resample 97M+ ticks to ~7M 1-sec bars in chunks (cheap aggregation),
then compute features on the compact bar array. This avoids chunk-boundary
corruption that would occur if computing features inside each tick chunk.

Uses 30-second windows (30 × 1-sec bars) with 6 features per window.

Usage:
    python scripts/precompute_micro_features_v2.py
    python scripts/precompute_micro_features_v2.py --years 2025-2026
    python scripts/precompute_micro_features_v2.py --years 2025 --window 30
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime.micro_features_v2 import (
    DEFAULT_MICRO_WINDOW,
    MES_TICK,
    MICRO_FEATURE_NAMES,
    micro_features_from_1s_bars,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
L1_DIR = DATA_DIR / "l1"
FEATURES_DIR = DATA_DIR / "features" / "micro_hmm_v2"


def _available_years() -> list[int]:
    if not L1_DIR.exists():
        return []
    years = []
    for p in sorted(L1_DIR.iterdir()):
        if p.name.startswith("year=") and (p / "data.parquet").exists():
            years.append(int(p.name.split("=")[1]))
    return years


def _resample_l1_to_1sec_bars(in_path: Path, chunk_size: int = 5_000_000) -> pd.DataFrame:
    """Resample L1 ticks to 1-sec bars with real spread and OFI.

    Processes ticks in chunks to keep peak RAM low, aggregates per-second
    partial results, then combines. This avoids chunk-boundary corruption
    because we aggregate partials at second resolution before concatenating.

    Returns DataFrame with columns: close, volume, spread_ticks, ofi, timestamp.
    """
    pf = pq.ParquetFile(in_path)
    # Only read the columns we need
    needed = ["timestamp", "bid_price", "ask_price", "bid_size", "ask_size", "size"]
    available = pf.schema_arrow.names
    cols = [c for c in needed if c in available]

    partials: list[pd.DataFrame] = []
    n_ticks = 0

    for batch in pf.iter_batches(batch_size=chunk_size, columns=cols):
        df = batch.to_pandas()
        n_ticks += len(df)

        bid = df["bid_price"].values.astype(np.float64)
        ask = df["ask_price"].values.astype(np.float64)
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / MES_TICK
        ts = df["timestamp"].values
        seconds = ts.astype("datetime64[s]")

        # OFI: delta(bid_size) - delta(ask_size)
        bid_sz = df["bid_size"].values.astype(np.float64) if "bid_size" in df.columns else np.ones(len(df))
        ask_sz = df["ask_size"].values.astype(np.float64) if "ask_size" in df.columns else np.ones(len(df))
        d_bid = np.diff(bid_sz, prepend=bid_sz[0])
        d_ask = np.diff(ask_sz, prepend=ask_sz[0])
        ofi = d_bid - d_ask

        volume = df["size"].values.astype(np.float64) if "size" in df.columns else np.ones(len(df))

        # Per-second partial aggregation (cheap — just groupby on int64 seconds)
        sec_int = seconds.astype(np.int64)
        chunk_df = pd.DataFrame({
            "sec": sec_int,
            "mid_last": mid,
            "volume": volume,
            "spread_sum": spread,
            "spread_count": np.ones(len(spread)),
            "ofi": ofi,
        })
        partial = chunk_df.groupby("sec").agg(
            mid_last=("mid_last", "last"),
            volume=("volume", "sum"),
            spread_sum=("spread_sum", "sum"),
            spread_count=("spread_count", "sum"),
            ofi=("ofi", "sum"),
        )
        partials.append(partial)
        del df, chunk_df, bid, ask, mid, spread, ofi, volume
        gc.collect()

        click.echo(f"    resampled {n_ticks:,} ticks -> {sum(len(p) for p in partials):,} partial bars")

    if not partials:
        return pd.DataFrame(columns=["close", "volume", "spread_ticks", "ofi", "timestamp"])

    # Combine partials — seconds that span chunk boundaries get merged
    combined = pd.concat(partials)
    del partials
    gc.collect()

    # Re-aggregate: for seconds that appeared in multiple chunks,
    # take last mid, sum volume/ofi, weighted-average spread
    bars = combined.groupby(combined.index).agg(
        close=("mid_last", "last"),
        volume=("volume", "sum"),
        spread_sum=("spread_sum", "sum"),
        spread_count=("spread_count", "sum"),
        ofi=("ofi", "sum"),
    )
    bars["spread_ticks"] = bars["spread_sum"] / bars["spread_count"]
    bars["timestamp"] = bars.index.values.astype("datetime64[s]")
    bars = bars[["close", "volume", "spread_ticks", "ofi", "timestamp"]].copy()

    click.echo(f"    final: {n_ticks:,} ticks -> {len(bars):,} 1-sec bars")
    return bars


@click.command()
@click.option("--years", default=None,
              help="Year range (e.g., '2025-2026') or comma-separated. Default: all available.")
@click.option("--window", default=DEFAULT_MICRO_WINDOW, type=int,
              help=f"Number of 1-sec bars per window. Default: {DEFAULT_MICRO_WINDOW} (30s).")
@click.option("--force", is_flag=True, help="Recompute even if features already exist.")
@click.option("--chunk-size", default=5_000_000, type=int,
              help="Number of ticks to read at a time (memory management).")
def main(years: str | None, window: int, force: bool, chunk_size: int) -> None:
    """Precompute micro HMM v2 features from L1 tick data."""
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
        click.echo("No L1 data found in data/l1/.")
        return

    click.echo(f"Years: {year_list}")
    click.echo(f"Window: {window} bars ({window}s)")
    click.echo(f"Features: {len(MICRO_FEATURE_NAMES)} ({', '.join(MICRO_FEATURE_NAMES)})")
    click.echo(f"Input: {L1_DIR}/ (raw L1 ticks)")
    click.echo(f"Output: {FEATURES_DIR}/")
    click.echo()

    total_rows = 0

    for year in year_list:
        in_path = L1_DIR / f"year={year}" / "data.parquet"
        out_dir = FEATURES_DIR / f"year={year}"
        out_path = out_dir / "data.parquet"

        if not in_path.exists():
            click.echo(f"  [SKIP] No L1 data for {year}")
            continue

        if out_path.exists() and not force:
            existing = pq.read_metadata(str(out_path)).num_rows
            click.echo(f"  [EXISTS] {year}: {existing:,} rows (use --force to recompute)")
            total_rows += existing
            continue

        t0 = time.time()
        click.echo(f"  {year}: resampling L1 ticks to 1-sec bars...")

        # Step 1: Resample ticks to 1-sec bars (handles chunk boundaries correctly)
        bars_df = _resample_l1_to_1sec_bars(in_path, chunk_size=chunk_size)

        if len(bars_df) == 0:
            click.echo(f"  [WARN] {year}: no bars")
            continue

        # Step 2: Compute micro features on the compact bar array
        click.echo(f"  {year}: computing micro features from {len(bars_df):,} bars...")
        feats, timestamps = micro_features_from_1s_bars(bars_df, window=window)
        del bars_df
        gc.collect()

        if len(feats) == 0:
            click.echo(f"  [WARN] {year}: no valid features")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        columns = {
            name: feats[:, i].astype(np.float64)
            for i, name in enumerate(MICRO_FEATURE_NAMES)
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
