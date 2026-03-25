"""Sweep entry/exit thresholds for rolling backtest.

Runs rolling inference ONCE, then tests many threshold combinations
on the same P(up) signal. Fast because inference is the expensive part.

Usage:
    python scripts/sweep_backtest.py --year 2026 --model-name fallback
    python scripts/sweep_backtest.py --year 2025 --model-name pair_3_0
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_backtest
from src.backtest.position_manager import PositionManagerConfig
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import _compute_features, _filter_rth, _resample_to_1sec, _z_score_normalize
from src.models.entry_model import EntryModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)


@click.command()
@click.option("--year", default=2026, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--model-name", default="fallback", type=str)
@click.option("--window-size", default=30, type=int)
@click.option("--hard-sl", default=12.0, type=float)
@click.option("--max-hold", default=300, type=int)
@click.option("--commission", default=0.59, type=float)
@click.option("--batch-size", default=4096, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    year: int,
    model_dir: str | None,
    model_name: str,
    window_size: int,
    hard_sl: float,
    max_hold: int,
    commission: float,
    batch_size: int,
    verbose: bool,
) -> None:
    """Sweep thresholds on a single model's rolling inference."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}"
    else:
        model_dir_path = Path(model_dir)

    model_path = model_dir_path / f"{model_name}.pt"
    if not model_path.exists():
        click.echo(f"ERROR: Model not found at {model_path}")
        return

    click.echo(f"Model: {model_path}")
    click.echo(f"Year: {year}, Device: {device}")
    click.echo()

    # ── Load data ─────────────────────────────────────────────────
    click.echo("[1/3] LOADING DATA...")
    l1_path = L1_DIR / f"year={year}" / "data.parquet"
    pf = pq.ParquetFile(l1_path)
    df = pf.read().to_pandas()
    click.echo(f"  {len(df):,} ticks")

    bars = _resample_to_1sec(df)
    del df
    gc.collect()

    bars = _filter_rth(bars)
    bar_seconds = bars.index.values
    gaps = np.diff(bar_seconds)
    session_breaks = np.where(gaps > 60)[0] + 1

    raw_features = _compute_features(bars)
    features = _z_score_normalize(raw_features)
    del raw_features
    gc.collect()

    mid = bars["mid"].values.astype(np.float32)
    bid = bars["bid"].values.astype(np.float32)
    ask = bars["ask"].values.astype(np.float32)
    del bars
    gc.collect()

    click.echo(f"  {len(features):,} bars, {len(session_breaks)} sessions")

    # ── Inference (once) ──────────────────────────────────────────
    click.echo("\n[2/3] ROLLING INFERENCE (one-time)...")
    model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    t0 = time.time()
    p_up = rolling_inference(model, features, window_size=window_size,
                             batch_size=batch_size, device=device)
    click.echo(f"  {time.time() - t0:.1f}s")

    valid_p = p_up[np.isfinite(p_up)]
    click.echo(f"  P(up) distribution:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        click.echo(f"    p{pct:2d} = {np.percentile(valid_p, pct):.4f}")

    del model, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Sweep ─────────────────────────────────────────────────────
    click.echo(f"\n[3/3] SWEEPING THRESHOLDS...")

    # Entry thresholds (symmetric around 0.5)
    entry_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    # Exit offsets from entry (how far signal must drop before exit)
    exit_offsets = [0.05, 0.10, 0.15, 0.20, 0.25]

    click.echo(f"\n  {'Entry':>5} | {'Exit_L':>6} | {'Exit_S':>6} | {'Trades':>7} | "
               f"{'Long':>5} | {'Short':>6} | {'WR':>5} | {'AvgPnL':>7} | "
               f"{'PF':>5} | {'Net$':>9} | {'MaxDD':>6} | {'AvgHold':>7}")
    click.echo(f"  {'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-"
               f"{'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}")

    best_net = float("-inf")
    best_params = None

    for entry_t in entry_thresholds:
        long_entry = 0.5 + (entry_t - 0.5)   # e.g., 0.70
        short_entry = 0.5 - (entry_t - 0.5)  # e.g., 0.30

        for exit_offset in exit_offsets:
            long_exit = long_entry - exit_offset    # e.g., 0.70 - 0.15 = 0.55 → exit when drops below
            short_exit = short_entry + exit_offset  # e.g., 0.30 + 0.15 = 0.45 → exit when rises above

            # Clamp to valid range
            long_exit = max(0.05, min(long_exit, long_entry - 0.01))
            short_exit = min(0.95, max(short_exit, short_entry + 0.01))

            config = PositionManagerConfig(
                long_entry=long_entry,
                short_entry=short_entry,
                long_exit=long_exit,
                short_exit=short_exit,
                hard_sl_ticks=hard_sl,
                max_hold_bars=max_hold,
                commission_rt_dollars=commission,
            )

            result = run_backtest(p_up, mid, bid, ask, session_breaks, config)
            r = result

            if r.n_trades == 0:
                continue

            click.echo(
                f"  {long_entry:.2f} | {long_exit:6.2f} | {short_exit:6.2f} | "
                f"{r.n_trades:>7,} | {r.n_long:>5,} | {r.n_short:>6,} | "
                f"{r.win_rate:>5.1%} | {r.avg_pnl_ticks:>+7.2f} | "
                f"{r.profit_factor:>5.2f} | {r.net_pnl_dollars:>+9.2f} | "
                f"{r.max_drawdown_ticks:>6.0f} | {r.avg_hold_bars:>7.0f}s"
            )

            if r.net_pnl_dollars > best_net:
                best_net = r.net_pnl_dollars
                best_params = {
                    "long_entry": long_entry,
                    "short_entry": short_entry,
                    "long_exit": long_exit,
                    "short_exit": short_exit,
                    "n_trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "avg_pnl_ticks": r.avg_pnl_ticks,
                    "profit_factor": r.profit_factor,
                    "net_pnl": r.net_pnl_dollars,
                    "max_dd_ticks": r.max_drawdown_ticks,
                    "avg_hold": r.avg_hold_bars,
                }

    if best_params:
        click.echo(f"\n  BEST: entry={best_params['long_entry']:.2f}/{best_params['short_entry']:.2f}, "
                   f"exit={best_params['long_exit']:.2f}/{best_params['short_exit']:.2f}")
        click.echo(f"    {best_params['n_trades']} trades, WR={best_params['win_rate']:.1%}, "
                   f"PF={best_params['profit_factor']:.2f}, Net=${best_params['net_pnl']:+,.2f}, "
                   f"MaxDD={best_params['max_dd_ticks']:.0f}t, Hold={best_params['avg_hold']:.0f}s")
    else:
        click.echo("\n  No trades generated at any threshold combination.")


if __name__ == "__main__":
    main()
