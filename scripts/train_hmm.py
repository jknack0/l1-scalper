"""Train HMM regime detector on precomputed MES features.

Loads precomputed 1-minute feature parquets from data/features/hmm/.
Run scripts/precompute_features.py first to generate them.

Usage:
    python scripts/train_hmm.py
    python scripts/train_hmm.py --train-years 2020-2024 --test-year 2025
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

from src.regime.features import FEATURE_NAMES
from src.regime.hmm import MarketRegimeHMM, STATE_NAMES
from src.regime.trainer import evaluate_regime_stability

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURES_DIR = DATA_DIR / "features" / "hmm"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_features(years: list[int]) -> tuple[np.ndarray, list[int]]:
    """Load precomputed feature parquets for given years.

    Returns:
        features: [total_windows, 4] concatenated array
        lengths: per-year row counts (for HMM sequence boundaries)
    """
    chunks: list[np.ndarray] = []
    lengths: list[int] = []

    for year in years:
        path = FEATURES_DIR / f"year={year}" / "data.parquet"
        if not path.exists():
            click.echo(f"  [SKIP] No precomputed features for {year}")
            continue

        t = pq.read_table(path)
        arr = np.column_stack([t.column(name).to_numpy() for name in FEATURE_NAMES])
        chunks.append(arr)
        lengths.append(len(arr))
        click.echo(f"  {year}: {len(arr):,} feature vectors")

    if not chunks:
        raise RuntimeError(
            f"No precomputed features found. Run: python scripts/precompute_features.py --years {years[0]}-{years[-1]}"
        )

    return np.concatenate(chunks, axis=0), lengths


@click.command()
@click.option("--train-years", default="2020-2024",
              help="Year range for training (e.g., '2020-2024').")
@click.option("--test-year", default=2025, type=int,
              help="Year for validation.")
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Model output path. Default: models/hmm_regime.pkl")
@click.option("--visualize/--no-visualize", default=True,
              help="Generate regime visualization plots.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    train_years: str,
    test_year: int,
    output: Path | None,
    visualize: bool,
    verbose: bool,
) -> None:
    """Train HMM regime detector on precomputed MES features."""
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

    output_path = output or MODEL_DIR / "hmm_regime.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load precomputed training features ────────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"[1/6] LOADING TRAINING FEATURES: years {years[0]}-{years[-1]}")
    click.echo(f"{'='*60}")

    step_t0 = time.time()
    train_features, train_lengths = _load_features(years)
    click.echo(f"  Total: {len(train_features):,} vectors in {time.time() - step_t0:.1f}s")

    # ── Step 2: Load precomputed test features ────────────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"[2/6] LOADING VALIDATION FEATURES: year {test_year}")
    click.echo(f"{'='*60}")

    step_t0 = time.time()
    test_features, _ = _load_features([test_year])
    click.echo(f"  Total: {len(test_features):,} test vectors in {time.time() - step_t0:.1f}s")
    ohlcv_test_features = None

    # ── Step 3: Feature statistics ────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[3/6] FEATURE STATISTICS (raw, before normalization)")
    click.echo(f"{'='*60}")

    click.echo("\n  Training set:")
    for i, name in enumerate(FEATURE_NAMES):
        col = train_features[:, i]
        click.echo(f"    {name:20s}: mean={col.mean():+.6f}  std={col.std():.6f}  "
                    f"min={col.min():+.6f}  max={col.max():+.6f}")

    click.echo("\n  Test set:")
    for i, name in enumerate(FEATURE_NAMES):
        col = test_features[:, i]
        click.echo(f"    {name:20s}: mean={col.mean():+.6f}  std={col.std():.6f}  "
                    f"min={col.min():+.6f}  max={col.max():+.6f}")

    # ── Step 4: Normalize + Train ─────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[4/6] Z-SCORE NORMALIZATION + HMM TRAINING")
    click.echo(f"{'='*60}")

    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std[train_std < 1e-10] = 1.0

    click.echo(f"  Training mean: [{', '.join(f'{m:+.6f}' for m in train_mean)}]")
    click.echo(f"  Training std:  [{', '.join(f'{s:.6f}' for s in train_std)}]")

    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std
    if ohlcv_test_features is not None:
        ohlcv_test_features = (ohlcv_test_features - train_mean) / train_std

    step_t0 = time.time()
    click.echo(f"\n  Training 3-state Gaussian HMM (300 EM iterations)...")
    click.echo(f"  Input shape: {train_features.shape}")
    click.echo(f"  Sequence lengths: {train_lengths}")
    model = MarketRegimeHMM(n_iter=300, random_state=42)
    model.fit(train_features, lengths=train_lengths)
    click.echo(f"  HMM trained in {time.time() - step_t0:.1f}s")

    # ── Step 5: Evaluate ──────────────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[5/6] EVALUATION")
    click.echo(f"{'='*60}")

    click.echo(f"\n  --- Training set regime report ---")
    train_report = evaluate_regime_stability(model, train_features)
    _print_report(train_report)

    click.echo(f"\n  --- Test set ({test_year}) regime report ---")
    test_report = evaluate_regime_stability(model, test_features)
    _print_report(test_report)

    if ohlcv_test_features is not None:
        click.echo(f"\n  --- Test set ({test_year} OHLCV) regime report ---")
        ohlcv_report = evaluate_regime_stability(model, ohlcv_test_features)
        _print_report(ohlcv_report)

    # Transition matrix
    click.echo(f"\n  --- Transition Matrix ---")
    tm = model.transition_matrix
    click.echo(f"  {'':20s} -> Trending  MeanRev   Choppy")
    for i, from_name in enumerate(["Trending", "MeanReverting", "Choppy"]):
        click.echo(f"    {from_name:18s}  {tm[i,0]:.4f}    {tm[i,1]:.4f}    {tm[i,2]:.4f}")

    # State emission means
    click.echo(f"\n  --- State Emission Means (normalized) ---")
    means = model.means
    click.echo(f"  {'State':18s} {'return_autocorr':>16s} {'hurst':>14s} "
               f"{'variance_ratio':>16s} {'efficiency_ratio':>18s}")
    for state_id, name in STATE_NAMES.items():
        m = means[state_id]
        click.echo(f"    {name:16s} {m[0]:16.6f} {m[1]:14.6f} {m[2]:16.6f} {m[3]:18.6f}")

    # ── Step 6: Save model + visualize ────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[6/6] SAVING MODEL & VISUALIZATIONS")
    click.echo(f"{'='*60}")

    model.save(output_path)
    click.echo(f"  Model saved to {output_path}")

    if visualize:
        click.echo("  Generating visualization plots...")

        from src.regime.visualizer import plot_regime_overlay
        from src.regime.bocpd import BOCPD

        plots_dir = MODEL_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Use a 1-week slice of test data for readable plots
        n_week = min(5 * 390, len(test_features))
        test_slice = test_features[:n_week]

        labels = model.predict(test_slice)
        posteriors = model.predict_proba_sequence(test_slice)

        # Run BOCPD on efficiency_ratio
        bocpd = BOCPD()
        cp_probs = np.array([bocpd.detect(float(x)) for x in test_slice[:, 3]])

        prices = test_slice[:, 3]  # efficiency_ratio

        plot_regime_overlay(
            prices=prices,
            regime_labels=labels,
            changepoint_probs=cp_probs,
            posteriors=posteriors,
            title=f"HMM Regimes — {test_year} (first week, efficiency_ratio)",
            output_path=plots_dir / f"regime_{test_year}_week1.png",
        )
        click.echo(f"  Saved: {plots_dir / f'regime_{test_year}_week1.png'}")

        # Full test set regime distribution over time
        full_labels = model.predict(test_features)
        full_posteriors = model.predict_proba_sequence(test_features)

        step = max(1, len(test_features) // 5000)
        plot_regime_overlay(
            prices=test_features[::step, 0],
            regime_labels=full_labels[::step],
            posteriors=full_posteriors[::step],
            title=f"HMM Regimes — {test_year} (full year, subsampled)",
            output_path=plots_dir / f"regime_{test_year}_full.png",
        )
        click.echo(f"  Saved: {plots_dir / f'regime_{test_year}_full.png'}")

    # ── Summary ───────────────────────────────────────────────────
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
