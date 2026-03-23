"""Offline HMM training on historical 1-minute bar features.

Loads Parquet feature data, resamples to 1-minute bars, fits HMM,
evaluates on held-out data, and saves the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq

from src.regime.hmm import MarketRegimeHMM, STATE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)


def resample_1s_to_1min(
    timestamps: np.ndarray,
    features: np.ndarray,
    feature_indices: list[int],
) -> np.ndarray:
    """Resample 1-second feature data to 1-minute bars using mean aggregation.

    Args:
        timestamps: shape [n] — nanosecond timestamps
        features: shape [n, num_features] — 1-second feature vectors
        feature_indices: indices of the 4 features to extract
            [realized_vol_idx, return_autocorr_idx, spread_idx, trade_rate_idx]

    Returns:
        shape [n_minutes, 4] — 1-minute mean-aggregated features
    """
    ONE_MIN_NS = 60_000_000_000
    minute_keys = timestamps // ONE_MIN_NS

    unique_minutes = np.unique(minute_keys)
    result = []

    for mk in unique_minutes:
        mask = minute_keys == mk
        bar_features = features[mask][:, feature_indices]
        if len(bar_features) > 0:
            result.append(bar_features.mean(axis=0))

    if not result:
        return np.empty((0, N_FEATURES))

    return np.array(result)


def train_hmm(
    data: np.ndarray,
    train_pct: float = 0.7,
    n_iter: int = 200,
    random_state: int = 42,
) -> tuple[MarketRegimeHMM, dict]:
    """Train HMM on data and evaluate on held-out portion.

    Args:
        data: shape [n_bars, 4]
        train_pct: fraction to use for training

    Returns:
        (fitted model, evaluation report dict)
    """
    n = data.shape[0]
    split = int(n * train_pct)
    train_data = data[:split]
    test_data = data[split:]

    logger.info("Training on %d bars, evaluating on %d bars", len(train_data), len(test_data))

    model = MarketRegimeHMM(n_iter=n_iter, random_state=random_state)
    model.fit(train_data)

    # Evaluate on test set
    report = evaluate_regime_stability(model, test_data)
    report["train_bars"] = len(train_data)
    report["test_bars"] = len(test_data)

    return model, report


def evaluate_regime_stability(
    model: MarketRegimeHMM,
    data: np.ndarray,
    state_names: dict[int, str] | None = None,
) -> dict:
    """Evaluate regime label stability on held-out data.

    Works with both macro (MarketRegimeHMM) and micro (MicroRegimeHMM) models.

    Returns:
        Dict with per-state time percentages, mean durations, transition matrix.
    """
    if state_names is None:
        state_names = STATE_NAMES

    labels = model.predict(data)
    n = len(labels)

    # Time in each state
    state_pcts = {}
    for state_id, name in state_names.items():
        pct = float(np.mean(labels == state_id))
        state_pcts[name] = round(pct * 100, 1)

    # Average duration per state (consecutive run lengths)
    durations: dict[int, list[int]] = {i: [] for i in state_names}
    run_len = 1
    for i in range(1, n):
        if labels[i] == labels[i - 1]:
            run_len += 1
        else:
            durations[labels[i - 1]].append(run_len)
            run_len = 1
    durations[labels[-1]].append(run_len)

    avg_durations = {}
    for state_id, name in state_names.items():
        runs = durations[state_id]
        avg_durations[name] = round(float(np.mean(runs)), 1) if runs else 0.0

    return {
        "state_pcts": state_pcts,
        "avg_durations_bars": avg_durations,
        "transition_matrix": model.transition_matrix.tolist(),
    }


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory with 1-second feature Parquet files.")
@click.option("--output", type=click.Path(path_type=Path), required=True,
              help="Path to save the trained HMM model.")
@click.option("--train-pct", type=float, default=0.7, help="Training split fraction.")
@click.option("--feature-indices", type=str, default="11,7,11,11",
              help="Comma-separated indices of [return_autocorr, hurst, variance_ratio, efficiency_ratio]. "
                   "Note: variance_ratio and efficiency_ratio are computed from returns, not stored features.")
@click.option("--verbose", "-v", is_flag=True)
def cli(
    data_dir: Path,
    output: Path,
    train_pct: float,
    feature_indices: str,
    verbose: bool,
) -> None:
    """Train HMM regime detector on historical feature data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    feat_idx = [int(x) for x in feature_indices.split(",")]
    assert len(feat_idx) == 4, "Need exactly 4 feature indices"

    # Load all Parquet files
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        click.echo(f"No parquet files found in {data_dir}")
        return

    click.echo(f"Loading {len(files)} files from {data_dir}")

    all_timestamps = []
    all_features = []

    for f in files:
        table = pq.read_table(f)
        df = table.to_pandas()
        if "timestamp" in df.columns and len(df) > 0:
            all_timestamps.append(df["timestamp"].values)
            # Assume feature columns are in order after timestamp
            feat_cols = [c for c in df.columns if c != "timestamp"]
            all_features.append(df[feat_cols].values)

    if not all_timestamps:
        click.echo("No data found")
        return

    timestamps = np.concatenate(all_timestamps)
    features = np.concatenate(all_features)

    click.echo(f"Loaded {len(timestamps)} seconds of data")

    # Resample to 1-minute
    minute_data = resample_1s_to_1min(timestamps, features, feat_idx)
    click.echo(f"Resampled to {len(minute_data)} 1-minute bars")

    # Train
    model, report = train_hmm(minute_data, train_pct=train_pct)

    # Save
    model.save(output)
    click.echo(f"Model saved to {output}")

    # Report
    click.echo("\n--- Regime Report ---")
    click.echo(f"Train bars: {report['train_bars']}, Test bars: {report['test_bars']}")
    click.echo("\nTime in each state (test set):")
    for name, pct in report["state_pcts"].items():
        click.echo(f"  {name}: {pct}%")
    click.echo("\nAvg duration per state (bars):")
    for name, dur in report["avg_durations_bars"].items():
        click.echo(f"  {name}: {dur}")
    click.echo(f"\nTransition matrix:\n{np.array2string(np.array(report['transition_matrix']), precision=3)}")


if __name__ == "__main__":
    cli()
