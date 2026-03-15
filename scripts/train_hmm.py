"""Train HMM regime detector on actual MES data.

Uses OHLCV 1-second bars (2011-2024) for training and L1 tick data (2025) for validation.

Computes 4 HMM features per 1-minute bar:
    - realized_vol: std of 1-second log returns over the 1-min bar
    - return_autocorr: lag-1 autocorrelation of 1-second returns in the bar
    - spread_mean: mean (high - low) per 1s bar as spread proxy (OHLCV) or
                   mean (ask - bid) in ticks (L1)
    - trade_rate_mean: mean volume per second in the bar (OHLCV) or
                       trade count per second (L1)

Usage:
    python scripts/train_hmm.py
    python scripts/train_hmm.py --train-years 2020-2024 --test-year 2025
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.regime.hmm import MarketRegimeHMM, STATE_NAMES
from src.regime.trainer import evaluate_regime_stability

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MES_TICK = 0.25
ONE_MIN_SECONDS = 60


def _vectorized_autocorr(returns: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Vectorized lag-1 autocorrelation per group — no groupby.apply().

    Computes cov(r_t, r_{t+1}) / (std(r_t) * std(r_{t+1})) per group
    using only vectorized pandas transforms.
    """
    # Build lag-1 pairs within each group
    r = returns
    r_lag = np.empty_like(r)
    r_lag[0] = np.nan

    # r_lag[i] = r[i-1] only if same group
    same_group = groups[1:] == groups[:-1]
    r_lag[1:] = np.where(same_group, r[:-1], np.nan)

    df = pd.DataFrame({"r": r, "rl": r_lag, "g": groups})
    df = df.dropna()

    if df.empty:
        return np.zeros(len(np.unique(groups)))

    # Per-group means via transform (vectorized)
    mean_r = df.groupby("g")["r"].transform("mean")
    mean_rl = df.groupby("g")["rl"].transform("mean")

    dr = df["r"].values - mean_r.values
    drl = df["rl"].values - mean_rl.values
    g_vals = df["g"].values

    # Per-group sums via bincount
    n_groups = int(g_vals.max()) + 1
    counts = np.bincount(g_vals, minlength=n_groups).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero

    sum_cov = np.bincount(g_vals, weights=dr * drl, minlength=n_groups)
    sum_dr2 = np.bincount(g_vals, weights=dr**2, minlength=n_groups)
    sum_drl2 = np.bincount(g_vals, weights=drl**2, minlength=n_groups)

    cov = sum_cov / counts
    std_r = np.sqrt(sum_dr2 / counts)
    std_rl = np.sqrt(sum_drl2 / counts)

    denom = std_r * std_rl
    result = np.where(denom > 1e-15, cov / denom, 0.0)

    # Return only for groups that appear in the aggregated output
    unique_groups = np.unique(groups)
    return result[unique_groups]


def features_from_ohlcv(df: pd.DataFrame) -> np.ndarray:
    """Compute HMM features from 1-second OHLCV bars, aggregated to 1-minute.

    Fully vectorized — no per-group Python loops.
    Returns shape [n_minutes, 4].
    """
    logger.info("  Computing OHLCV features (vectorized)...")

    close = df["close"].values.astype(np.float64)
    log_ret = np.log(close[1:] / close[:-1])

    spread_proxy = ((df["high"].values - df["low"].values) / MES_TICK)[1:]
    volume = df["volume"].values[1:].astype(np.float64)
    minutes = df["timestamp"].values[1:].astype("datetime64[m]")

    # Remove non-finite
    mask = np.isfinite(log_ret)
    log_ret = log_ret[mask]
    spread_proxy = spread_proxy[mask]
    volume = volume[mask]
    minutes = minutes[mask]

    # Group by minute — use numpy for speed
    unique_mins, inverse, counts = np.unique(minutes, return_inverse=True, return_counts=True)

    # Filter groups with < 10 samples
    valid_groups = counts >= 10
    valid_group_mask = valid_groups[inverse]

    log_ret = log_ret[valid_group_mask]
    spread_proxy = spread_proxy[valid_group_mask]
    volume = volume[valid_group_mask]
    minutes = minutes[valid_group_mask]

    # Re-index
    unique_mins, inverse = np.unique(minutes, return_inverse=True)
    n_groups = len(unique_mins)

    logger.info("  Aggregating %d 1-sec bars into %d 1-min bars...", len(log_ret), n_groups)

    # Realized vol = per-group std
    df_tmp = pd.DataFrame({"ret": log_ret, "spread": spread_proxy, "vol": volume, "g": inverse})
    agg = df_tmp.groupby("g").agg(
        realized_vol=("ret", "std"),
        spread_mean=("spread", "mean"),
        trade_rate=("vol", "mean"),
    )

    # Autocorrelation (vectorized)
    autocorr = _vectorized_autocorr(log_ret, inverse)
    agg["return_autocorr"] = autocorr

    result = agg[["realized_vol", "return_autocorr", "spread_mean", "trade_rate"]].values
    mask = np.all(np.isfinite(result), axis=1)
    return result[mask]


def features_from_l1(df: pd.DataFrame) -> np.ndarray:
    """Compute HMM features from L1 tick data, aggregated to 1-minute.

    Vectorized: first resample ticks to 1-second, then aggregate to 1-minute.
    Returns shape [n_minutes, 4].
    """
    logger.info("  Computing L1 features (vectorized)...")

    bid = df["bid_price"].values.astype(np.float64)
    ask = df["ask_price"].values.astype(np.float64)
    mid = (bid + ask) / 2.0
    spread_ticks = (ask - bid) / MES_TICK
    ts = df["timestamp"].values

    # Floor to second and minute
    seconds = ts.astype("datetime64[s]")
    minutes_raw = ts.astype("datetime64[m]")

    logger.info("  Resampling %d ticks to 1-second bars...", len(mid))

    # Resample to 1-second: last mid, mean spread, count trades
    sec_df = pd.DataFrame({
        "mid": mid, "spread": spread_ticks, "second": seconds, "minute": minutes_raw,
    })
    sec_agg = sec_df.groupby("second").agg(
        mid=("mid", "last"),
        spread=("spread", "mean"),
        trade_count=("mid", "count"),
        minute=("minute", "first"),
    )

    logger.info("  Got %d 1-second bars, computing 1-min features...", len(sec_agg))

    # Log returns of 1-second mid prices
    mids = sec_agg["mid"].values
    log_ret = np.log(mids[1:] / mids[:-1])
    spreads = sec_agg["spread"].values[1:]
    trade_counts = sec_agg["trade_count"].values[1:].astype(np.float64)
    min_keys = sec_agg["minute"].values[1:]

    # Remove non-finite
    mask = np.isfinite(log_ret)
    log_ret = log_ret[mask]
    spreads = spreads[mask]
    trade_counts = trade_counts[mask]
    min_keys = min_keys[mask]

    # Group by minute
    unique_mins, inverse, counts = np.unique(min_keys, return_inverse=True, return_counts=True)
    valid = counts >= 10
    valid_mask = valid[inverse]

    log_ret = log_ret[valid_mask]
    spreads = spreads[valid_mask]
    trade_counts = trade_counts[valid_mask]
    min_keys = min_keys[valid_mask]
    unique_mins, inverse = np.unique(min_keys, return_inverse=True)

    logger.info("  Aggregating %d seconds into %d 1-min bars...", len(log_ret), len(unique_mins))

    df_tmp = pd.DataFrame({
        "ret": log_ret, "spread": spreads, "tc": trade_counts, "g": inverse,
    })
    agg = df_tmp.groupby("g").agg(
        realized_vol=("ret", "std"),
        spread_mean=("spread", "mean"),
        trade_rate=("tc", "mean"),
    )

    autocorr = _vectorized_autocorr(log_ret, inverse)
    agg["return_autocorr"] = autocorr

    result = agg[["realized_vol", "return_autocorr", "spread_mean", "trade_rate"]].values
    mask = np.all(np.isfinite(result), axis=1)
    return result[mask]


def load_ohlcv_years(years: list[int]) -> pd.DataFrame:
    """Load OHLCV parquet data for specified years."""
    frames = []
    for year in years:
        path = DATA_DIR / "parquet" / f"year={year}" / "data.parquet"
        if not path.exists():
            logger.warning("No OHLCV data for year %d", year)
            continue
        logger.info("Loading OHLCV %d...", year)
        df = pq.read_table(path).to_pandas()
        frames.append(df)
        logger.info("  %d rows loaded", len(df))

    if not frames:
        raise RuntimeError("No OHLCV data found")

    return pd.concat(frames, ignore_index=True)


def load_l1_year(year: int) -> pd.DataFrame:
    """Load L1 tick data for a year."""
    path = DATA_DIR / "l1" / f"year={year}" / "data.parquet"
    if not path.exists():
        raise RuntimeError(f"No L1 data for year {year}")
    logger.info("Loading L1 %d...", year)
    df = pq.read_table(path).to_pandas()
    logger.info("  %d rows loaded", len(df))
    return df


@click.command()
@click.option("--train-years", default="2020-2024",
              help="Year range for training (e.g., '2020-2024' or '2011-2024').")
@click.option("--test-year", default=2025, type=int,
              help="Year for validation (uses L1 data if available, else OHLCV).")
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
    """Train HMM regime detector on MES historical data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Parse year range
    if "-" in train_years:
        start_yr, end_yr = train_years.split("-")
        years = list(range(int(start_yr), int(end_yr) + 1))
    else:
        years = [int(y) for y in train_years.split(",")]

    output_path = output or MODEL_DIR / "hmm_regime.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract training features from OHLCV ────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"TRAINING: OHLCV years {years[0]}-{years[-1]}")
    click.echo(f"{'='*60}")

    train_df = load_ohlcv_years(years)
    click.echo(f"Computing features from {len(train_df):,} OHLCV bars...")
    train_features = features_from_ohlcv(train_df)
    click.echo(f"Extracted {len(train_features):,} 1-minute feature vectors")

    # Free memory
    del train_df

    # ── Step 2: Extract test features ───────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo(f"VALIDATION: Year {test_year}")
    click.echo(f"{'='*60}")

    l1_path = DATA_DIR / "l1" / f"year={test_year}" / "data.parquet"
    if l1_path.exists():
        click.echo("Using L1 tick data for validation (more accurate features)")
        test_df = load_l1_year(test_year)
        click.echo(f"Computing features from {len(test_df):,} L1 ticks...")
        test_features = features_from_l1(test_df)

        # Also compute OHLCV features for comparison
        ohlcv_path = DATA_DIR / "parquet" / f"year={test_year}" / "data.parquet"
        if ohlcv_path.exists():
            ohlcv_test_df = pq.read_table(ohlcv_path).to_pandas()
            ohlcv_test_features = features_from_ohlcv(ohlcv_test_df)
            click.echo(f"Also extracted {len(ohlcv_test_features):,} features from OHLCV for comparison")
            del ohlcv_test_df
        else:
            ohlcv_test_features = None

        del test_df
    else:
        click.echo("No L1 data, using OHLCV for validation")
        test_df = load_ohlcv_years([test_year])
        test_features = features_from_ohlcv(test_df)
        ohlcv_test_features = None
        del test_df

    click.echo(f"Extracted {len(test_features):,} 1-minute feature vectors")

    # ── Step 3: Feature statistics ──────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("FEATURE STATISTICS")
    click.echo(f"{'='*60}")
    feat_names = ["realized_vol", "return_autocorr", "spread_mean", "trade_rate"]

    click.echo("\nTraining set:")
    for i, name in enumerate(feat_names):
        col = train_features[:, i]
        click.echo(f"  {name:20s}: mean={col.mean():.6f}  std={col.std():.6f}  "
                    f"min={col.min():.6f}  max={col.max():.6f}")

    click.echo("\nTest set:")
    for i, name in enumerate(feat_names):
        col = test_features[:, i]
        click.echo(f"  {name:20s}: mean={col.mean():.6f}  std={col.std():.6f}  "
                    f"min={col.min():.6f}  max={col.max():.6f}")

    # ── Step 3b: Normalize features ────────────────────────────
    # Z-score normalize using training set statistics so the HMM
    # works consistently across OHLCV and L1 feature sources
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_std[train_std < 1e-10] = 1.0  # avoid division by zero

    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std
    if ohlcv_test_features is not None:
        ohlcv_test_features = (ohlcv_test_features - train_mean) / train_std

    click.echo("\nFeatures z-score normalized using training set statistics.")

    # ── Step 4: Train HMM ──────────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("TRAINING HMM")
    click.echo(f"{'='*60}")

    model = MarketRegimeHMM(n_iter=300, random_state=42)
    model.fit(train_features)

    # ── Step 5: Evaluate on training data ───────────────────────
    click.echo(f"\n--- Training set regime report ---")
    train_report = evaluate_regime_stability(model, train_features)
    _print_report(train_report)

    # ── Step 6: Evaluate on test data ───────────────────────────
    click.echo(f"\n--- Test set ({test_year}) regime report ---")
    test_report = evaluate_regime_stability(model, test_features)
    _print_report(test_report)

    if ohlcv_test_features is not None:
        click.echo(f"\n--- Test set ({test_year} OHLCV) regime report ---")
        ohlcv_report = evaluate_regime_stability(model, ohlcv_test_features)
        _print_report(ohlcv_report)

    # ── Step 7: Save model ──────────────────────────────────────
    model.save(output_path)
    click.echo(f"\nModel saved to {output_path}")

    # ── Step 8: Transition matrix ───────────────────────────────
    click.echo(f"\n--- Transition Matrix ---")
    tm = model.transition_matrix
    click.echo(f"{'':20s} -> Trending  MeanRev   Choppy")
    for i, from_name in enumerate(["Trending", "MeanReverting", "Choppy"]):
        click.echo(f"  {from_name:18s}  {tm[i,0]:.4f}    {tm[i,1]:.4f}    {tm[i,2]:.4f}")

    # ── Step 9: State means ────────────────────────────────────
    click.echo(f"\n--- State Emission Means ---")
    means = model.means
    click.echo(f"{'State':18s} {'realized_vol':>14s} {'return_autocorr':>16s} "
               f"{'spread_mean':>14s} {'trade_rate':>14s}")
    for state_id, name in STATE_NAMES.items():
        m = means[state_id]
        click.echo(f"  {name:16s} {m[0]:14.6f} {m[1]:16.6f} {m[2]:14.6f} {m[3]:14.6f}")

    # ── Step 10: Visualize ──────────────────────────────────────
    if visualize:
        click.echo(f"\n{'='*60}")
        click.echo("GENERATING VISUALIZATIONS")
        click.echo(f"{'='*60}")

        from src.regime.visualizer import plot_regime_overlay
        from src.regime.bocpd import BOCPD

        plots_dir = MODEL_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Use a 1-week slice of test data for readable plots
        n_week = min(5 * 390, len(test_features))  # 5 trading days * 390 min
        test_slice = test_features[:n_week]

        labels = model.predict(test_slice)
        posteriors = model.predict_proba_sequence(test_slice)

        # Run BOCPD on realized_vol
        bocpd = BOCPD()
        cp_probs = np.array([bocpd.detect(float(x)) for x in test_slice[:, 0]])

        # Use realized_vol as price proxy for visualization
        # (we don't have raw prices at this point, but vol shows regime structure)
        prices = test_slice[:, 0]  # realized_vol

        plot_regime_overlay(
            prices=prices,
            regime_labels=labels,
            changepoint_probs=cp_probs,
            posteriors=posteriors,
            title=f"HMM Regimes — {test_year} (first week, realized_vol)",
            output_path=plots_dir / f"regime_{test_year}_week1.png",
        )
        click.echo(f"  Saved: {plots_dir / f'regime_{test_year}_week1.png'}")

        # Full test set regime distribution over time
        full_labels = model.predict(test_features)
        full_posteriors = model.predict_proba_sequence(test_features)

        # Subsample for full-year plot (every 10th bar)
        step = max(1, len(test_features) // 5000)
        plot_regime_overlay(
            prices=test_features[::step, 0],
            regime_labels=full_labels[::step],
            posteriors=full_posteriors[::step],
            title=f"HMM Regimes — {test_year} (full year, subsampled)",
            output_path=plots_dir / f"regime_{test_year}_full.png",
        )
        click.echo(f"  Saved: {plots_dir / f'regime_{test_year}_full.png'}")

    click.echo(f"\n{'='*60}")
    click.echo("DONE")
    click.echo(f"{'='*60}")


def _print_report(report: dict) -> None:
    """Pretty-print a regime evaluation report."""
    click.echo("  Time in each state:")
    for name, pct in report["state_pcts"].items():
        click.echo(f"    {name:18s}: {pct:5.1f}%")
    click.echo("  Avg duration (bars):")
    for name, dur in report["avg_durations_bars"].items():
        click.echo(f"    {name:18s}: {dur:5.1f}")


if __name__ == "__main__":
    main()
