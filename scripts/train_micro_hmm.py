"""Train micro HMM regime detector on precomputed micro features.

Loads precomputed 1-sec micro feature parquets from data/features/micro_hmm/.
Run scripts/precompute_micro_features.py first.

Usage:
    python scripts/train_micro_hmm.py
    python scripts/train_micro_hmm.py --train-years 2020-2024 --test-year 2025
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime.micro_features import MICRO_FEATURE_NAMES
from src.regime.micro_hmm import MICRO_STATE_NAMES, MicroRegimeHMM
from src.regime.trainer import evaluate_regime_stability

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURES_DIR = DATA_DIR / "features" / "micro_hmm"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_features(years: list[int]) -> tuple[np.ndarray, list[int]]:
    """Load precomputed micro feature parquets for given years."""
    chunks: list[np.ndarray] = []
    lengths: list[int] = []

    for year in years:
        path = FEATURES_DIR / f"year={year}" / "data.parquet"
        if not path.exists():
            click.echo(f"  [SKIP] No precomputed micro features for {year}")
            continue

        t = pq.read_table(path)
        arr = np.column_stack([t.column(name).to_numpy() for name in MICRO_FEATURE_NAMES])
        chunks.append(arr)
        lengths.append(len(arr))
        click.echo(f"  {year}: {len(arr):,} feature vectors")

    if not chunks:
        raise RuntimeError(
            f"No precomputed micro features found. "
            f"Run: python scripts/precompute_micro_features.py --years {years[0]}-{years[-1]}"
        )

    return np.concatenate(chunks, axis=0), lengths


@click.command()
@click.option("--train-years", default="2024",
              help="Year range for training (e.g., '2020-2024').")
@click.option("--test-year", default=2025, type=int,
              help="Year for validation.")
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Model output path. Default: models/micro_hmm_regime.pkl")
@click.option("--verbose", "-v", is_flag=True)
def main(
    train_years: str,
    test_year: int,
    output: Path | None,
    verbose: bool,
) -> None:
    """Train micro HMM regime detector on precomputed micro features."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overall_t0 = time.time()

    # Parse year range
    if "-" in train_years:
        start_yr, end_yr = train_years.split("-")
        years = list(range(int(start_yr), int(end_yr) + 1))
    else:
        years = [int(y) for y in train_years.split(",")]

    output_path = output or MODEL_DIR / "micro_hmm_regime.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load training features ──────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"[1/5] LOADING TRAINING FEATURES: years {years[0]}-{years[-1]}")
    click.echo(f"{'='*60}")

    step_t0 = time.time()
    train_features, train_lengths = _load_features(years)
    click.echo(f"  Total: {len(train_features):,} vectors in {time.time() - step_t0:.1f}s")

    # ── Step 2: Load test features ──────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"[2/5] LOADING VALIDATION FEATURES: year {test_year}")
    click.echo(f"{'='*60}")

    step_t0 = time.time()
    try:
        test_features, _ = _load_features([test_year])
        click.echo(f"  Total: {len(test_features):,} test vectors in {time.time() - step_t0:.1f}s")
    except RuntimeError:
        click.echo(f"  [SKIP] No test data for {test_year}, will skip evaluation")
        test_features = None

    # ── Step 3: Feature statistics ──────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[3/5] FEATURE STATISTICS (raw)")
    click.echo(f"{'='*60}")

    click.echo("\n  Training set:")
    for i, name in enumerate(MICRO_FEATURE_NAMES):
        col = train_features[:, i]
        click.echo(f"    {name:20s}: mean={col.mean():+.6f}  std={col.std():.6f}  "
                    f"min={col.min():+.6f}  max={col.max():+.6f}")

    if test_features is not None:
        click.echo("\n  Test set:")
        for i, name in enumerate(MICRO_FEATURE_NAMES):
            col = test_features[:, i]
            click.echo(f"    {name:20s}: mean={col.mean():+.6f}  std={col.std():.6f}  "
                        f"min={col.min():+.6f}  max={col.max():+.6f}")

    # ── Step 4: Normalize + Train ───────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[4/5] Z-SCORE NORMALIZATION + MICRO HMM TRAINING")
    click.echo(f"{'='*60}")

    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std[train_std < 1e-10] = 1.0

    click.echo(f"  Training mean: [{', '.join(f'{m:+.6f}' for m in train_mean)}]")
    click.echo(f"  Training std:  [{', '.join(f'{s:.6f}' for s in train_std)}]")

    train_normed = (train_features - train_mean) / train_std

    step_t0 = time.time()
    click.echo(f"\n  Training 3-state Micro GMM-HMM (200 EM iterations)...")
    click.echo(f"  Input shape: {train_normed.shape}")
    click.echo(f"  Sequence lengths: {train_lengths}")
    model = MicroRegimeHMM(n_iter=200, random_state=42)
    model.fit(train_normed, lengths=train_lengths)
    click.echo(f"  Micro HMM trained in {time.time() - step_t0:.1f}s")

    # ── Step 5: Evaluate ────────────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[5/5] EVALUATION")
    click.echo(f"{'='*60}")

    click.echo(f"\n  --- Training set regime report ---")
    train_report = evaluate_regime_stability(model, train_normed, state_names=MICRO_STATE_NAMES)
    _print_report(train_report)

    if test_features is not None:
        test_normed = (test_features - train_mean) / train_std
        click.echo(f"\n  --- Test set ({test_year}) regime report ---")
        test_report = evaluate_regime_stability(model, test_normed, state_names=MICRO_STATE_NAMES)
        _print_report(test_report)

    # Transition matrix
    click.echo(f"\n  --- Transition Matrix ---")
    tm = model.transition_matrix
    click.echo(f"  {'':20s} -> LiqTrend  LiqMR     Illiquid")
    names = ["LiquidTrending", "LiquidMeanRev", "Illiquid"]
    for i, from_name in enumerate(names):
        click.echo(f"    {from_name:18s}  {tm[i,0]:.4f}    {tm[i,1]:.4f}    {tm[i,2]:.4f}")

    # State emission means
    click.echo(f"\n  --- State Emission Means (normalized) ---")
    means = model.means
    click.echo(f"  {'State':18s} {'spread':>10s} {'trade_rate':>12s} "
               f"{'autocorr':>10s} {'vol':>10s} {'ofi':>10s}")
    for state_id, name in MICRO_STATE_NAMES.items():
        m = means[state_id]
        click.echo(f"    {name:16s} {m[0]:10.4f} {m[1]:12.4f} {m[2]:10.4f} {m[3]:10.6f} {m[4]:10.4f}")

    # Save
    model.save(output_path)
    click.echo(f"\n  Model saved to {output_path}")

    total_elapsed = time.time() - overall_t0
    click.echo(f"\n{'='*60}")
    click.echo(f"DONE — total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    click.echo(f"{'='*60}")


def _print_report(report: dict) -> None:
    """Pretty-print a regime evaluation report."""
    click.echo("    Time in each state:")
    for name, pct in report["state_pcts"].items():
        click.echo(f"      {name:18s}: {pct:5.1f}%")
    click.echo("    Avg duration (bars):")
    for name, dur in report["avg_durations_bars"].items():
        click.echo(f"      {name:18s}: {dur:5.1f}")


if __name__ == "__main__":
    main()
