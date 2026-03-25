"""Validate macro HMM regime states.

Fits macro HMM on training years (OHLCV data, 2011+), labels held-out
validation year using forward-only filtering, reports state distributions
and per-state forward-return predictability.

Usage:
    python scripts/validate_macro.py
    python scripts/validate_macro.py --train-years 2011-2023 --val-years 2024
    python scripts/validate_macro.py --n-states 3,4,5
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

from src.regime.macro_features_v2 import DEFAULT_MACRO_WINDOW, MACRO_FEATURE_NAMES
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "regime_validation"
MACRO_FEATURES_DIR = DATA_DIR / "features" / "macro_hmm_v2"
OHLCV_DIR = DATA_DIR / "parquet"

MES_TICK = 0.25

logger = logging.getLogger(__name__)


def _parse_years(s: str) -> list[int]:
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in s.split(",")]


def _load_features(years: list[int]) -> np.ndarray:
    all_feats = []
    for year in years:
        path = MACRO_FEATURES_DIR / f"year={year}" / "data.parquet"
        if not path.exists():
            click.echo(f"  [SKIP] No macro features for {year}")
            continue
        t = pq.read_table(path)
        feats = np.column_stack([t.column(c).to_numpy() for c in MACRO_FEATURE_NAMES])
        all_feats.append(feats)
        click.echo(f"  Loaded {year}: {len(feats):,} windows")
    if not all_feats:
        return np.empty((0, len(MACRO_FEATURE_NAMES)))
    return np.concatenate(all_feats)


def _load_close(years: list[int]) -> np.ndarray:
    all_close = []
    for year in years:
        path = OHLCV_DIR / f"year={year}" / "data.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["close"])
        all_close.append(t.column("close").to_numpy().astype(np.float64))
    if not all_close:
        return np.empty(0)
    return np.concatenate(all_close)


@click.command()
@click.option("--train-years", default="2011-2023",
              help="Years for HMM training.")
@click.option("--val-years", default="2024",
              help="Years for validation (held-out).")
@click.option("--n-states", default="3,4,5",
              help="Comma-separated N values for BIC selection.")
@click.option("--n-mix", default=3, type=int, help="Gaussian mixture components per state.")
@click.option("--cov-type", default="full", type=click.Choice(["full", "diag", "spherical"]),
              help="Covariance type.")
@click.option("--verbose", "-v", is_flag=True)
def main(train_years: str, val_years: str, n_states: str, n_mix: int, cov_type: str, verbose: bool) -> None:
    """Fit macro HMM and validate states on held-out data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()

    train_list = _parse_years(train_years)
    val_list = _parse_years(val_years)
    candidates = [int(x) for x in n_states.split(",")]

    click.echo(f"Train years: {train_list}")
    click.echo(f"Val years:   {val_list}")
    click.echo(f"N candidates: {candidates}")
    click.echo()

    # ── Fit ────────────────────────────────────────────────────────
    click.echo("=" * 60)
    click.echo("[1/3] FITTING MACRO HMM")
    click.echo("=" * 60)

    train_feats = _load_features(train_list)
    if len(train_feats) == 0:
        click.echo("ERROR: No training features. Run precompute_macro_features_v2.py first.")
        return

    click.echo(f"  Training on {len(train_feats):,} windows")
    click.echo(f"  n_mix={n_mix}, cov_type={cov_type}")

    hmm = MacroRegimeHMMv2(n_mix=n_mix, covariance_type=cov_type)
    bic_results = hmm.fit_with_bic(train_feats, n_states_candidates=candidates)

    click.echo(f"\n  Selected: {hmm.n_states} states")
    for r in bic_results:
        click.echo(f"    N={r.n_states}: BIC={r.bic:.1f}, converged={r.converged}")

    hmm_path = MODEL_DIR / "macro_hmm_v2.pkl"
    hmm.save(hmm_path)
    del train_feats
    gc.collect()

    # ── Label validation data ─────────────────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[2/3] LABELING VALIDATION DATA (FORWARD-ONLY)")
    click.echo("=" * 60)

    val_feats = _load_features(val_list)
    if len(val_feats) == 0:
        click.echo("ERROR: No validation features.")
        return

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

    close = _load_close(val_list)
    if len(close) == 0:
        click.echo("ERROR: No 1-sec bar data for validation years.")
        return

    macro_window = DEFAULT_MACRO_WINDOW  # 300
    n_windows = len(close) // macro_window
    n_windows = min(n_windows, len(labels))

    # Forward returns: from end of each macro window, 5s and 60s ahead
    window_ends = np.arange(n_windows) * macro_window + (macro_window - 1)
    valid_5 = window_ends + 5 < len(close)
    valid_60 = window_ends + 60 < len(close)
    valid = valid_5 & valid_60
    window_ends = window_ends[valid]
    state_labels = labels[:n_windows][valid]

    ret_5s = (close[window_ends + 5] - close[window_ends]) / MES_TICK
    ret_60s = (close[window_ends + 60] - close[window_ends]) / MES_TICK

    click.echo(f"\n  {len(window_ends):,} windows with valid forward returns\n")

    # Per-state stats
    click.echo(f"  {'State':>5} | {'N':>8} | {'%':>5} | {'mean5s':>8} | {'std5s':>8} | "
               f"{'mean60s':>8} | {'std60s':>8} | {'autocorr':>8} | {'dir_rate':>8}")
    click.echo(f"  {'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for s in range(hmm.n_states):
        mask = state_labels == s
        n = mask.sum()
        if n < 10:
            continue
        pct = n / len(state_labels) * 100
        r5 = ret_5s[mask]
        r60 = ret_60s[mask]

        # Lag-1 autocorrelation of 5s returns
        if len(r5) > 2:
            autocorr = float(np.corrcoef(r5[:-1], r5[1:])[0, 1])
        else:
            autocorr = 0.0

        # Directional consistency: sign(5s) == sign(60s)
        s5 = np.sign(r5)
        s60 = np.sign(r60)
        nz = (s5 != 0) & (s60 != 0)
        dir_rate = float((s5[nz] == s60[nz]).mean()) if nz.sum() > 10 else 0.5

        click.echo(
            f"  {s:>5} | {n:>8,} | {pct:>5.1f} | {r5.mean():>8.3f} | {r5.std():>8.3f} | "
            f"{r60.mean():>8.3f} | {r60.std():>8.3f} | {autocorr:>8.4f} | {dir_rate:>8.3f}"
        )

    # Feature means per state (denormalized)
    click.echo(f"\n  Weighted emission means (denormalized):")
    means = hmm.weighted_means  # [n_states, n_features] — these are in normalized space
    # Denormalize for interpretability
    for s in range(hmm.n_states):
        denorm = means[s] * hmm._train_std + hmm._train_mean
        parts = [f"{MACRO_FEATURE_NAMES[i]}={denorm[i]:.6f}" for i in range(len(MACRO_FEATURE_NAMES))]
        click.echo(f"    State {s}: {', '.join(parts)}")

    elapsed = time.time() - t0
    click.echo(f"\nDone — {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    click.echo(f"Model saved: {hmm_path}")


if __name__ == "__main__":
    main()
