"""Backtest regime-gated models with rolling inference + signal-based exits.

Loads trained regime-specific models, runs rolling inference on test data,
and simulates trades using the position manager.

Usage:
    python scripts/backtest_rolling.py --year 2026
    python scripts/backtest_rolling.py --year 2025 --long-entry 0.70 --short-entry 0.30
    python scripts/backtest_rolling.py --year 2025 --long-exit 0.55 --hard-sl 15
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import print_result, run_backtest
from src.backtest.position_manager import PositionManagerConfig
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import _compute_features, _resample_to_1sec, _z_score_normalize, _filter_rth
from src.models.entry_model import EntryModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)


def _load_l1_bars(year: int, rth_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load L1 data, resample to 1-sec bars, compute features.

    Returns (features, mid, bid, ask, session_breaks).
    """
    l1_path = L1_DIR / f"year={year}" / "data.parquet"
    click.echo(f"  Loading L1 data for {year}...")

    pf = pq.ParquetFile(l1_path)
    df = pf.read().to_pandas()
    click.echo(f"  {len(df):,} ticks loaded")

    click.echo("  Resampling to 1-sec bars...")
    bars = _resample_to_1sec(df)
    del df
    gc.collect()
    click.echo(f"  {len(bars):,} 1-sec bars")

    if rth_only:
        bars = _filter_rth(bars)
        click.echo(f"  {len(bars):,} bars after RTH filter")

    # Session breaks
    bar_seconds = bars.index.values
    gaps = np.diff(bar_seconds)
    session_breaks = np.where(gaps > 60)[0] + 1

    # Features
    click.echo("  Computing features...")
    raw_features = _compute_features(bars)

    click.echo("  Normalizing...")
    norm_features = _z_score_normalize(raw_features)
    del raw_features
    gc.collect()

    mid = bars["mid"].values.astype(np.float32)
    bid = bars["bid"].values.astype(np.float32)
    ask = bars["ask"].values.astype(np.float32)

    del bars
    gc.collect()

    return norm_features, mid, bid, ask, session_breaks


@click.command()
@click.option("--year", default=2026, type=int, help="Year to backtest.")
@click.option("--model-dir", default=None, type=str,
              help="Directory with trained .pt models. Default: models/regime_v2_fold2/w30/")
@click.option("--model-name", default="fallback", type=str,
              help="Which model to use (e.g., 'fallback', 'pair_3_0', 'pair_4_0').")
@click.option("--window-size", default=30, type=int)
@click.option("--long-entry", default=0.70, type=float, help="P(up) threshold for long entry.")
@click.option("--short-entry", default=0.30, type=float, help="P(up) threshold for short entry.")
@click.option("--long-exit", default=0.50, type=float, help="P(up) threshold for long exit.")
@click.option("--short-exit", default=0.50, type=float, help="P(up) threshold for short exit.")
@click.option("--hard-sl", default=12.0, type=float, help="Hard stop loss in ticks.")
@click.option("--max-hold", default=300, type=int, help="Max hold time in bars (seconds).")
@click.option("--commission", default=0.59, type=float, help="Round-trip commission in dollars.")
@click.option("--batch-size", default=4096, type=int, help="Inference batch size.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    year: int,
    model_dir: str | None,
    model_name: str,
    window_size: int,
    long_entry: float,
    short_entry: float,
    long_exit: float,
    short_exit: float,
    hard_sl: float,
    max_hold: int,
    commission: float,
    batch_size: int,
    verbose: bool,
) -> None:
    """Backtest a trained model with rolling inference and signal-based exits."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model path
    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}"
    else:
        model_dir_path = Path(model_dir)

    model_path = model_dir_path / f"{model_name}.pt"
    if not model_path.exists():
        click.echo(f"ERROR: Model not found at {model_path}")
        click.echo(f"Available models: {[p.stem for p in model_dir_path.glob('*.pt')]}")
        return

    click.echo(f"{'=' * 60}")
    click.echo("ROLLING BACKTEST")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Model: {model_path}")
    click.echo(f"  Year: {year}")
    click.echo(f"  Window: {window_size}")
    click.echo(f"  Entry: long >= {long_entry}, short <= {short_entry}")
    click.echo(f"  Exit:  long < {long_exit}, short > {short_exit}")
    click.echo(f"  Hard SL: {hard_sl} ticks, Max hold: {max_hold}s")
    click.echo(f"  Device: {device}")
    click.echo()

    # ── Load data ─────────────────────────────────────────────────
    click.echo("[1/3] LOADING DATA")
    features, mid, bid, ask, session_breaks = _load_l1_bars(year)
    n_bars = len(features)
    n_features = features.shape[1]
    click.echo(f"  {n_bars:,} bars, {n_features} features, {len(session_breaks)} sessions")

    # ── Load model and run inference ──────────────────────────────
    click.echo(f"\n[2/3] ROLLING INFERENCE")
    model = EntryModel(n_features=n_features, seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    click.echo(f"  Model loaded: {model.count_params():,} params")

    t0 = time.time()
    p_up = rolling_inference(model, features, window_size=window_size,
                             batch_size=batch_size, device=device)
    inference_time = time.time() - t0
    click.echo(f"  Inference: {inference_time:.1f}s for {n_bars:,} bars "
               f"({n_bars / inference_time:,.0f} bars/sec)")

    # P(up) distribution
    valid_p = p_up[np.isfinite(p_up)]
    click.echo(f"  P(up) stats: mean={valid_p.mean():.3f}, std={valid_p.std():.3f}, "
               f"min={valid_p.min():.3f}, max={valid_p.max():.3f}")
    click.echo(f"  P(up) >= {long_entry}: {(valid_p >= long_entry).sum():,} bars "
               f"({(valid_p >= long_entry).mean():.1%})")
    click.echo(f"  P(up) <= {short_entry}: {(valid_p <= short_entry).sum():,} bars "
               f"({(valid_p <= short_entry).mean():.1%})")

    del model, features
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Run backtest ──────────────────────────────────────────────
    click.echo(f"\n[3/3] SIMULATING TRADES")

    config = PositionManagerConfig(
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        hard_sl_ticks=hard_sl,
        max_hold_bars=max_hold,
        commission_rt_dollars=commission,
    )

    result = run_backtest(
        p_up=p_up,
        mid=mid,
        bid=bid,
        ask=ask,
        session_breaks=session_breaks,
        config=config,
    )

    print_result(result)

    elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
