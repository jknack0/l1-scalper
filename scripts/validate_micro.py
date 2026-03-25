"""Validate micro HMM regime states.

Since L1 data only exists for 2025-2026, this script splits a single year
temporally: train on the first portion, validate on the rest.

Usage:
    python scripts/validate_micro.py
    python scripts/validate_micro.py --year 2025 --train-frac 0.7
    python scripts/validate_micro.py --n-states 3,4,5
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime.micro_features_v2 import DEFAULT_MICRO_WINDOW, MICRO_FEATURE_NAMES
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "regime_validation"
MICRO_FEATURES_DIR = DATA_DIR / "features" / "micro_hmm_v2"
OHLCV_DIR = DATA_DIR / "parquet"

MES_TICK = 0.25

logger = logging.getLogger(__name__)


def _load_micro_features(year: int) -> np.ndarray:
    path = MICRO_FEATURES_DIR / f"year={year}" / "data.parquet"
    if not path.exists():
        click.echo(f"  ERROR: No micro features for {year} at {path}")
        return np.empty((0, len(MICRO_FEATURE_NAMES)))
    t = pq.read_table(path)
    feats = np.column_stack([t.column(c).to_numpy() for c in MICRO_FEATURE_NAMES])
    click.echo(f"  Loaded {year}: {len(feats):,} windows")
    return feats


def _load_close(year: int) -> np.ndarray:
    path = OHLCV_DIR / f"year={year}" / "data.parquet"
    if not path.exists():
        return np.empty(0)
    t = pq.read_table(path, columns=["close"])
    return t.column("close").to_numpy().astype(np.float64)


@click.command()
@click.option("--year", default=2025, type=int,
              help="Year to use (split temporally).")
@click.option("--train-frac", default=0.7, type=float,
              help="Fraction of data for HMM training (rest is validation).")
@click.option("--n-states", default="3,4,5",
              help="Comma-separated N values for BIC selection.")
@click.option("--verbose", "-v", is_flag=True)
def main(year: int, train_frac: float, n_states: str, verbose: bool) -> None:
    """Fit micro HMM and validate states via intra-year temporal split."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()

    candidates = [int(x) for x in n_states.split(",")]

    click.echo(f"Year: {year}")
    click.echo(f"Train fraction: {train_frac:.0%}")
    click.echo(f"N candidates: {candidates}")
    click.echo()

    # ── Load features ─────────────────────────────────────────────
    feats = _load_micro_features(year)
    if len(feats) == 0:
        return

    n = len(feats)
    split = int(n * train_frac)
    train_feats = feats[:split]
    val_feats = feats[split:]

    click.echo(f"  Train: {len(train_feats):,} windows (first {train_frac:.0%})")
    click.echo(f"  Val:   {len(val_feats):,} windows (last {1-train_frac:.0%})")
    click.echo()

    # ── Fit ────────────────────────────────────────────────────────
    click.echo("=" * 60)
    click.echo("[1/3] FITTING MICRO HMM")
    click.echo("=" * 60)

    hmm = MicroRegimeHMMv2()
    bic_results = hmm.fit_with_bic(train_feats, n_states_candidates=candidates)

    click.echo(f"\n  Selected: {hmm.n_states} states")
    for r in bic_results:
        click.echo(f"    N={r.n_states}: BIC={r.bic:.1f}, converged={r.converged}")

    hmm_path = MODEL_DIR / "micro_hmm_v2.pkl"
    hmm.save(hmm_path)

    # ── Label validation data (forward-only) ──────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[2/3] LABELING VALIDATION DATA (FORWARD-ONLY)")
    click.echo("=" * 60)

    val_norm = hmm.normalize(val_feats)
    posteriors = hmm.predict_proba_forward(val_norm)
    labels = posteriors.argmax(axis=1)

    for s in range(hmm.n_states):
        count = (labels == s).sum()
        pct = count / len(labels) * 100
        click.echo(f"  State {s}: {count:,} windows ({pct:.1f}%)")

    # ── Per-state forward return analysis ─────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[3/3] PER-STATE FORWARD RETURN ANALYSIS")
    click.echo("=" * 60)

    close = _load_close(year)
    if len(close) == 0:
        click.echo("ERROR: No 1-sec bar data.")
        return

    # Align: val windows start at bar index = split * micro_window
    micro_window = DEFAULT_MICRO_WINDOW  # 30
    val_start_bar = split * micro_window
    val_close = close[val_start_bar:]

    n_val_windows = min(len(val_close) // micro_window, len(labels))
    window_ends = np.arange(n_val_windows) * micro_window + (micro_window - 1)

    # Forward returns at 5s and 30s horizons
    valid_5 = window_ends + 5 < len(val_close)
    valid_30 = window_ends + 30 < len(val_close)
    valid = valid_5 & valid_30
    window_ends = window_ends[valid]
    state_labels = labels[:n_val_windows][valid]

    ret_5s = (val_close[window_ends + 5] - val_close[window_ends]) / MES_TICK
    ret_30s = (val_close[window_ends + 30] - val_close[window_ends]) / MES_TICK

    click.echo(f"\n  {len(window_ends):,} windows with valid forward returns\n")

    click.echo(f"  {'State':>5} | {'N':>8} | {'%':>5} | {'mean5s':>8} | {'std5s':>8} | "
               f"{'mean30s':>8} | {'std30s':>8} | {'autocorr':>8} | {'dir_rate':>8}")
    click.echo(f"  {'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for s in range(hmm.n_states):
        mask = state_labels == s
        n_s = mask.sum()
        if n_s < 10:
            continue
        pct = n_s / len(state_labels) * 100
        r5 = ret_5s[mask]
        r30 = ret_30s[mask]

        if len(r5) > 2:
            autocorr = float(np.corrcoef(r5[:-1], r5[1:])[0, 1])
        else:
            autocorr = 0.0

        s5 = np.sign(r5)
        s30 = np.sign(r30)
        nz = (s5 != 0) & (s30 != 0)
        dir_rate = float((s5[nz] == s30[nz]).mean()) if nz.sum() > 10 else 0.5

        click.echo(
            f"  {s:>5} | {n_s:>8,} | {pct:>5.1f} | {r5.mean():>8.3f} | {r5.std():>8.3f} | "
            f"{r30.mean():>8.3f} | {r30.std():>8.3f} | {autocorr:>8.4f} | {dir_rate:>8.3f}"
        )

    # Feature means per state
    click.echo(f"\n  Weighted emission means (denormalized):")
    means = hmm.weighted_means
    for s in range(hmm.n_states):
        denorm = means[s] * hmm._train_std + hmm._train_mean
        parts = [f"{MICRO_FEATURE_NAMES[i]}={denorm[i]:.4f}" for i in range(len(MICRO_FEATURE_NAMES))]
        click.echo(f"    State {s}: {', '.join(parts)}")

    elapsed = time.time() - t0
    click.echo(f"\nDone — {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    click.echo(f"Model saved: {hmm_path}")


if __name__ == "__main__":
    main()
