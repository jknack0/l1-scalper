"""Validate regime pairs before training entry NNs.

Fits macro + micro HMMs on training data, labels held-out data using
forward-only filtering, then runs predictability scoring per (macro, micro)
pair. Outputs a JSON validation report.

This script should be run BEFORE train_entry_regime_v2.py to verify
regimes are useful.

Usage:
    python scripts/validate_regimes.py
    python scripts/validate_regimes.py --hmm-train-years 2011-2023 --val-years 2024
    python scripts/validate_regimes.py --macro-states 3,4,5 --micro-states 3,4,5
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

from src.regime.macro_features_v2 import MACRO_FEATURE_NAMES
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_features_v2 import MICRO_FEATURE_NAMES
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2
from src.regime.regime_validator import save_report, validate_regime_pairs

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "regime_validation"

MACRO_FEATURES_DIR = DATA_DIR / "features" / "macro_hmm_v2"
MICRO_FEATURES_DIR = DATA_DIR / "features" / "micro_hmm_v2"
OHLCV_DIR = DATA_DIR / "parquet"

logger = logging.getLogger(__name__)


def _parse_years(s: str) -> list[int]:
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in s.split(",")]


def _load_features(feat_dir: Path, years: list[int], feat_names: list[str]) -> np.ndarray:
    """Load precomputed features for given years."""
    all_feats = []
    for year in years:
        path = feat_dir / f"year={year}" / "data.parquet"
        if not path.exists():
            click.echo(f"  [SKIP] No features for {year} at {path}")
            continue
        t = pq.read_table(path)
        feats = np.column_stack([t.column(c).to_numpy() for c in feat_names])
        all_feats.append(feats)
        click.echo(f"  Loaded {year}: {len(feats):,} windows")
    if not all_feats:
        return np.empty((0, len(feat_names)))
    return np.concatenate(all_feats)


def _load_1s_bars_for_forward_returns(years: list[int]) -> np.ndarray:
    """Load 1-sec close prices for computing forward returns."""
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


def _compute_forward_returns(
    close: np.ndarray,
    macro_window: int,
    micro_window: int,
    horizon_5s: int = 5,
    horizon_15s: int = 15,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute forward returns aligned to micro windows.

    Each micro window covers `micro_window` 1-sec bars. For each window,
    compute the return from the window's last bar to horizon bars ahead.

    Returns (ret_5s, ret_15s, n_windows) where n_windows is the number
    of micro windows that have enough future data for both horizons.
    """
    n_bars = len(close)
    n_micro = n_bars // micro_window

    if n_micro == 0:
        return np.empty(0), np.empty(0), 0

    # Window end indices (last bar of each micro window)
    window_ends = np.arange(1, n_micro + 1) * micro_window - 1

    # Only keep windows where we have enough future bars
    max_horizon = max(horizon_5s, horizon_15s)
    valid = window_ends + max_horizon < n_bars
    window_ends = window_ends[valid]
    n_valid = len(window_ends)

    if n_valid == 0:
        return np.empty(0), np.empty(0), 0

    ret_5s = close[window_ends + horizon_5s] - close[window_ends]
    ret_15s = close[window_ends + horizon_15s] - close[window_ends]

    return ret_5s, ret_15s, n_valid


@click.command()
@click.option("--hmm-train-years", default="2011-2023",
              help="Years for HMM training (before gap).")
@click.option("--val-years", default="2024",
              help="Years for validation (held-out from HMM).")
@click.option("--macro-states", default="3,4,5",
              help="Comma-separated N values for macro BIC selection.")
@click.option("--micro-states", default="3,4,5",
              help="Comma-separated N values for micro BIC selection.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    hmm_train_years: str,
    val_years: str,
    macro_states: str,
    micro_states: str,
    verbose: bool,
) -> None:
    """Validate regime pairs for NN training worthiness."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overall_t0 = time.time()

    train_years = _parse_years(hmm_train_years)
    val_year_list = _parse_years(val_years)
    macro_n_candidates = [int(x) for x in macro_states.split(",")]
    micro_n_candidates = [int(x) for x in micro_states.split(",")]

    click.echo(f"HMM train years: {train_years}")
    click.echo(f"Validation years: {val_year_list}")
    click.echo(f"Macro state candidates: {macro_n_candidates}")
    click.echo(f"Micro state candidates: {micro_n_candidates}")
    click.echo()

    # ── Step 1: Load and fit macro HMM ────────────────────────────
    click.echo("=" * 60)
    click.echo("[1/5] FITTING MACRO HMM")
    click.echo("=" * 60)

    macro_train = _load_features(MACRO_FEATURES_DIR, train_years, MACRO_FEATURE_NAMES)
    if len(macro_train) == 0:
        click.echo("ERROR: No macro training features found. Run precompute_macro_features_v2.py first.")
        return

    click.echo(f"  Macro training data: {len(macro_train):,} windows")

    macro_hmm = MacroRegimeHMMv2()
    macro_bic_results = macro_hmm.fit_with_bic(
        macro_train, n_states_candidates=macro_n_candidates,
    )
    click.echo(f"  Selected: {macro_hmm.n_states} states")
    for r in macro_bic_results:
        click.echo(f"    N={r.n_states}: BIC={r.bic:.1f}, converged={r.converged}")

    # Save macro HMM
    macro_hmm_path = MODEL_DIR / "macro_hmm_v2.pkl"
    macro_hmm.save(macro_hmm_path)
    del macro_train
    gc.collect()

    # ── Step 2: Load and fit micro HMM ────────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[2/5] FITTING MICRO HMM")
    click.echo("=" * 60)

    micro_train = _load_features(MICRO_FEATURES_DIR, train_years, MICRO_FEATURE_NAMES)
    if len(micro_train) == 0:
        click.echo("ERROR: No micro training features found. Run precompute_micro_features_v2.py first.")
        return

    click.echo(f"  Micro training data: {len(micro_train):,} windows")

    micro_hmm = MicroRegimeHMMv2()
    micro_bic_results = micro_hmm.fit_with_bic(
        micro_train, n_states_candidates=micro_n_candidates,
    )
    click.echo(f"  Selected: {micro_hmm.n_states} states")
    for r in micro_bic_results:
        click.echo(f"    N={r.n_states}: BIC={r.bic:.1f}, converged={r.converged}")

    micro_hmm_path = MODEL_DIR / "micro_hmm_v2.pkl"
    micro_hmm.save(micro_hmm_path)
    del micro_train
    gc.collect()

    # ── Step 3: Label validation data (forward-only) ──────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[3/5] LABELING VALIDATION DATA (FORWARD-ONLY)")
    click.echo("=" * 60)

    macro_val = _load_features(MACRO_FEATURES_DIR, val_year_list, MACRO_FEATURE_NAMES)
    micro_val = _load_features(MICRO_FEATURES_DIR, val_year_list, MICRO_FEATURE_NAMES)

    if len(macro_val) == 0 or len(micro_val) == 0:
        click.echo("ERROR: No validation features. Run precompute scripts for validation years.")
        return

    # Normalize using training stats (stored in HMM)
    macro_val_norm = macro_hmm.normalize(macro_val)
    micro_val_norm = micro_hmm.normalize(micro_val)

    # Forward-only filtering (NO Viterbi)
    click.echo("  Running forward-only filtering on macro...")
    macro_posteriors = macro_hmm.predict_proba_forward(macro_val_norm)
    macro_labels = macro_posteriors.argmax(axis=1)

    click.echo("  Running forward-only filtering on micro...")
    micro_posteriors = micro_hmm.predict_proba_forward(micro_val_norm)
    micro_labels = micro_posteriors.argmax(axis=1)

    # Log state distributions
    for s in range(macro_hmm.n_states):
        pct = (macro_labels == s).mean() * 100
        click.echo(f"  Macro state {s}: {(macro_labels == s).sum():,} ({pct:.1f}%)")
    for s in range(micro_hmm.n_states):
        pct = (micro_labels == s).mean() * 100
        click.echo(f"  Micro state {s}: {(micro_labels == s).sum():,} ({pct:.1f}%)")

    # ── Step 4: Align labels and compute forward returns ──────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[4/5] COMPUTING FORWARD RETURNS")
    click.echo("=" * 60)

    # Align: micro windows are 30-sec, macro windows are 300-sec
    # Each macro window spans 10 micro windows
    macro_window = 300
    micro_window = 30
    ratio = macro_window // micro_window  # 10

    # Expand macro labels to micro resolution
    macro_labels_micro = np.repeat(macro_labels, ratio)
    # Trim to match micro labels length
    n_common = min(len(macro_labels_micro), len(micro_labels))
    macro_labels_aligned = macro_labels_micro[:n_common]
    micro_labels_aligned = micro_labels[:n_common]

    click.echo(f"  Aligned {n_common:,} micro windows with macro labels")

    # Load 1-sec close prices for forward returns
    close = _load_1s_bars_for_forward_returns(val_year_list)
    if len(close) == 0:
        click.echo("ERROR: No 1-sec bar data for validation years.")
        return

    ret_5s, ret_15s, n_valid = _compute_forward_returns(
        close, macro_window, micro_window,
    )
    click.echo(f"  Forward returns: {n_valid:,} windows with valid 5s+15s horizons")

    # Trim labels to match forward returns
    n_final = min(n_common, n_valid)
    macro_labels_final = macro_labels_aligned[:n_final]
    micro_labels_final = micro_labels_aligned[:n_final]
    ret_5s = ret_5s[:n_final]
    ret_15s = ret_15s[:n_final]

    # Simple features for baseline model: use micro features (normalized)
    micro_val_for_baseline = micro_val_norm[:n_final]

    # ── Step 5: Run validation ────────────────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[5/5] VALIDATING REGIME PAIRS")
    click.echo("=" * 60)

    report = validate_regime_pairs(
        macro_labels=macro_labels_final,
        micro_labels=micro_labels_final,
        forward_returns_5s=ret_5s,
        forward_returns_15s=ret_15s,
        features=micro_val_for_baseline,
        macro_n_states=macro_hmm.n_states,
        micro_n_states=micro_hmm.n_states,
        macro_hmm_means=macro_hmm.weighted_means,
        micro_hmm_means=micro_hmm.weighted_means,
    )

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "validation_report.json"
    save_report(report, report_path)

    # Summary
    click.echo(f"\n{'=' * 60}")
    click.echo("SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Macro states: {report.macro_n_states}")
    click.echo(f"  Micro states: {report.micro_n_states}")
    click.echo(f"  Total pairs: {len(report.pairs)}")
    click.echo(f"  Tradeable: {len(report.tradeable_pairs)} ({', '.join(report.tradeable_pairs)})")
    click.echo(f"  Merge: {len(report.merge_map)} ({', '.join(f'{k}->{v}' for k, v in report.merge_map.items())})")
    click.echo(f"  Skip: {len(report.skip_pairs)} ({', '.join(report.skip_pairs)})")

    click.echo(f"\n  Report saved: {report_path}")

    for pair in sorted(report.pairs, key=lambda p: p.n_samples, reverse=True):
        click.echo(
            f"  ({pair.macro_state},{pair.micro_state}): "
            f"n={pair.n_samples:,}, "
            f"baseline={pair.baseline_accuracy:.3f}, "
            f"mag={pair.mean_magnitude_ticks:.2f}t, "
            f"consist={pair.directional_consistency:.3f} "
            f"-> {pair.classification}"
            f"{f' (merge->{pair.merge_target})' if pair.merge_target else ''}"
        )

    elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
