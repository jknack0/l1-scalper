"""Adaptive stop parameter sweep with walk-forward validation.

Random search over AdaptiveStopConfig parameter space per regime pair.
Evaluates via walk-forward: optimize on expanding train, test on next month.

Usage:
    python scripts/sweep_adaptive_stop.py
    python scripts/sweep_adaptive_stop.py --year 2025 --n-samples 500
    python scripts/sweep_adaptive_stop.py --models pair_3_0 pair_1_0
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.position_manager import AdaptiveStopConfig, MES_TICK, MES_TICK_VALUE
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import (
    _compute_features, _filter_rth, _resample_to_1sec, _z_score_normalize,
)
from src.models.entry_model import EntryModel
from src.regime.macro_features_v2 import macro_features_from_1s_bars
from src.regime.micro_features_v2 import micro_features_from_1s_bars

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "adaptive_stop_sweep"

logger = logging.getLogger(__name__)

COMMISSION_RT = 0.59
MAX_HOLD = 300
MIN_TRAIN_TRADES = 10
MIN_OOS_TRADES = 5

# Z-score entry thresholds (symmetric: long if z >= threshold, short if z <= -threshold)
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Rolling z-score lookback
ZSCORE_LOOKBACK = 300

# Walk-forward settings
WF_MIN_TRAIN_BARS = 50_000

PARAM_RANGES = {
    "hard_sl_ticks": [4, 6, 8, 10, 12],
    "breakeven_trigger_ticks": [0, 2, 3, 4, 6],
    "breakeven_lock_ticks": [0.5, 1.0, 1.5, 2.0],
    "tier1_activation_ticks": [0, 2, 3, 4, 6],
    "tier1_trail_distance": [1.0, 1.5, 2.0, 3.0, 4.0],
    "tier2_activation_ticks": [0, 5, 6, 8, 10],
    "tier2_trail_distance": [0.5, 1.0, 1.5, 2.0],
    "tier3_activation_ticks": [0, 8, 10, 12, 15],
    "tier3_trail_distance": [0.25, 0.5, 1.0],
    "velocity_lookback_bars": [0, 2, 3, 5, 10],
    "velocity_threshold_ticks": [3, 4, 6, 8],
    "velocity_trail_distance": [0.25, 0.5, 1.0],
}


def sample_valid_config(rng: np.random.Generator) -> AdaptiveStopConfig:
    """Sample a random valid AdaptiveStopConfig.

    Enforces all constraints:
    - Tier activations strictly increasing (when non-zero)
    - Tier distances strictly decreasing (when active)
    - Disabled tier disables higher tiers
    - Sub-params only sampled when parent mechanic active
    """
    pick = lambda key: rng.choice(PARAM_RANGES[key])

    hard_sl = float(pick("hard_sl_ticks"))

    be_trigger = float(pick("breakeven_trigger_ticks"))
    be_lock = float(pick("breakeven_lock_ticks")) if be_trigger > 0 else 0.0

    t1_act = float(pick("tier1_activation_ticks"))
    t1_dist = float(pick("tier1_trail_distance")) if t1_act > 0 else 2.0

    if t1_act > 0:
        t2_act_choices = [v for v in PARAM_RANGES["tier2_activation_ticks"]
                          if v == 0 or v > t1_act]
        t2_act = float(rng.choice(t2_act_choices)) if t2_act_choices else 0.0
    else:
        t2_act = 0.0

    if t2_act > 0:
        t2_dist_choices = [v for v in PARAM_RANGES["tier2_trail_distance"]
                           if v < t1_dist]
        t2_dist = float(rng.choice(t2_dist_choices)) if t2_dist_choices else t1_dist / 2
    else:
        t2_dist = 1.0

    if t2_act > 0:
        t3_act_choices = [v for v in PARAM_RANGES["tier3_activation_ticks"]
                          if v == 0 or v > t2_act]
        t3_act = float(rng.choice(t3_act_choices)) if t3_act_choices else 0.0
    else:
        t3_act = 0.0

    if t3_act > 0:
        t3_dist_choices = [v for v in PARAM_RANGES["tier3_trail_distance"]
                           if v < t2_dist]
        t3_dist = float(rng.choice(t3_dist_choices)) if t3_dist_choices else t2_dist / 2
    else:
        t3_dist = 0.5

    vel_lookback = int(pick("velocity_lookback_bars"))
    vel_threshold = float(pick("velocity_threshold_ticks")) if vel_lookback > 0 else 0.0
    vel_trail = float(pick("velocity_trail_distance")) if vel_lookback > 0 else 0.5

    return AdaptiveStopConfig(
        hard_sl_ticks=hard_sl,
        breakeven_trigger_ticks=be_trigger,
        breakeven_lock_ticks=be_lock,
        tier1_activation_ticks=t1_act,
        tier1_trail_distance=t1_dist,
        tier2_activation_ticks=t2_act,
        tier2_trail_distance=t2_dist,
        tier3_activation_ticks=t3_act,
        tier3_trail_distance=t3_dist,
        velocity_lookback_bars=vel_lookback,
        velocity_threshold_ticks=vel_threshold,
        velocity_trail_distance=vel_trail,
        max_hold_bars=MAX_HOLD,
        commission_rt_dollars=COMMISSION_RT,
    )


# -- Z-score computation -----------------------------------------------

def _rolling_zscore(p_up: np.ndarray, lookback: int = ZSCORE_LOOKBACK) -> np.ndarray:
    """Rolling z-score of P(up). Returns NaN where insufficient data."""
    n = len(p_up)
    z = np.full(n, np.nan, dtype=np.float32)
    valid = np.isfinite(p_up)
    p_clean = np.where(valid, p_up, 0.0)
    cs = np.cumsum(p_clean)
    cs2 = np.cumsum(p_clean ** 2)
    cv = np.cumsum(valid.astype(np.float64))
    min_samples = min(lookback, 120)
    for i in range(min_samples, n):
        if not valid[i]:
            continue
        j = max(0, i - lookback)
        count = cv[i] - cv[j]
        if count < 30:
            continue
        s = cs[i] - cs[j]
        s2 = cs2[i] - cs2[j]
        mean = s / count
        var = s2 / count - mean ** 2
        if var < 1e-10:
            continue
        z[i] = (p_up[i] - mean) / np.sqrt(var)
    return z


def _run_zscore_backtest(
    z_scores: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
    entry_z: float,
    cfg: AdaptiveStopConfig,
    min_hold: int = 30,
) -> BacktestResult:
    """Backtest with z-score entry + adaptive stop exit.

    Entry: z >= entry_z (long), z <= -entry_z (short)
    Exit: managed by AdaptiveStopConfig logic
    """
    from src.backtest.position_manager import PositionManager, Side, Trade

    n = len(z_scores)
    session_set = set(session_breaks) if session_breaks is not None else set()

    # Use a PositionManager with the adaptive config but override entry logic
    pm = PositionManager(cfg)
    cooldown_until = 0

    for i in range(n):
        # Session boundary
        if i in session_set and not pm.is_flat:
            pm.force_close(i, 0.5, mid[i])
            pm.reset()
            cooldown_until = i + min_hold

        if not pm.is_flat:
            # Let adaptive stop manage exit (pass dummy p_up since we don't use signal exit)
            pm.update(i, 0.5, mid[i])
        else:
            if i < cooldown_until:
                continue
            z = z_scores[i]
            if np.isnan(z):
                continue
            # Entry via z-score
            if z >= entry_z:
                pm._open(i, z, mid[i], Side.LONG)
            elif z <= -entry_z:
                pm._open(i, z, mid[i], Side.SHORT)

    # Force close remaining
    if not pm.is_flat:
        pm.force_close(n - 1, 0.5, mid[n - 1])

    trades = pm.trades

    # Apply realistic fills
    for t in trades:
        if t.side == Side.LONG:
            t.pnl_ticks = (bid[t.exit_bar] - ask[t.entry_bar]) / MES_TICK
        else:
            t.pnl_ticks = (bid[t.entry_bar] - ask[t.exit_bar]) / MES_TICK

    # Build BacktestResult
    from src.backtest.engine import _compute_stats
    return _compute_stats(trades, cfg.commission_rt_dollars)


# -- Scoring ----------------------------------------------------------

def _score(result: BacktestResult) -> float:
    """Score: net_pnl / (1 + max_drawdown_ticks). -inf if too few trades."""
    if result.n_trades < MIN_TRAIN_TRADES:
        return float("-inf")
    return result.net_pnl_dollars / (1.0 + result.max_drawdown_ticks)


# -- Sweep fold -------------------------------------------------------

def _sweep_fold(
    z_scores: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[tuple[AdaptiveStopConfig, float] | None, float, list[dict]]:
    """Random search on a train fold using z-score entry."""
    best_config = None
    best_entry_z = 0.0
    best_score = float("-inf")
    all_results: list[dict] = []
    total_iters = n_samples
    completed = 0
    t0 = time.time()

    samples_per_entry = max(1, n_samples // len(ENTRY_THRESHOLDS))

    for entry_z in ENTRY_THRESHOLDS:
        for _ in range(samples_per_entry):
            cfg = sample_valid_config(rng)

            result = _run_zscore_backtest(
                z_scores, mid, bid, ask, session_breaks,
                entry_z=entry_z, cfg=cfg,
            )
            score = _score(result)

            row = asdict(cfg)
            row["entry_z"] = entry_z
            row["n_trades"] = result.n_trades
            row["win_rate"] = result.win_rate
            row["net_pnl_dollars"] = result.net_pnl_dollars
            row["max_drawdown_ticks"] = result.max_drawdown_ticks
            row["profit_factor"] = result.profit_factor
            row["score"] = score
            all_results.append(row)

            if score > best_score:
                best_score = score
                best_config = cfg
                best_entry_z = entry_z

            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_iters - completed) / rate if rate > 0 else 0
                logger.info("      sweep progress: %d/%d (%.1f/s, ETA %.0fs) best_score=%.2f",
                            completed, total_iters, rate, eta, best_score)

    if best_config is not None:
        return (best_config, best_entry_z), best_score, all_results
    return None, best_score, all_results


# -- Data helpers (copied from sweep_walkforward.py) -------------------

def _load_bars(year: int, chunk_size: int = 10_000_000) -> pd.DataFrame:
    """Load L1 data in chunks, resample to 1-sec bars, filter to RTH.

    Processes ticks in chunks to keep peak RAM manageable (~2-3GB instead of 8GB+).
    """
    l1_path = L1_DIR / f"year={year}" / "data.parquet"
    click.echo(f"  Loading L1 data for {year} (chunked)...")
    pf = pq.ParquetFile(l1_path)

    all_bars = []
    n_ticks = 0

    for batch in pf.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        n_ticks += len(df_chunk)
        click.echo(f"    processed {n_ticks:,} ticks...")

        chunk_bars = _resample_to_1sec(df_chunk)
        all_bars.append(chunk_bars)
        del df_chunk
        gc.collect()

    click.echo(f"  {n_ticks:,} total ticks")

    # Concatenate and re-aggregate bars that span chunk boundaries
    bars = pd.concat(all_bars)
    del all_bars
    gc.collect()

    # Dedup: same second might appear in two chunks — keep last
    bars = bars[~bars.index.duplicated(keep='last')]
    bars = bars.sort_index()

    bars = _filter_rth(bars)
    click.echo(f"  {len(bars):,} RTH 1-sec bars")
    return bars


def _compute_regime_labels(
    bars: pd.DataFrame,
    macro_hmm_path: Path,
    micro_hmm_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bar (macro_state, micro_state) using forward-only HMM.

    Returns arrays of shape [n_bars] with state labels.
    Macro updates every 300 bars, micro every 30 bars. Labels are held
    constant between updates (same as live bot).
    """
    from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
    from src.regime.micro_hmm_v2 import MicroRegimeHMMv2

    n_bars = len(bars)

    # Load HMMs
    macro_hmm = MacroRegimeHMMv2()
    macro_hmm.load(macro_hmm_path)
    micro_hmm = MicroRegimeHMMv2()
    micro_hmm.load(micro_hmm_path)

    # Compute macro features (300-bar windows)
    macro_bars = pd.DataFrame({
        "close": bars["mid"].values,
        "volume": bars["total_vol"].values,
    }, index=bars.index)
    macro_feats, _ = macro_features_from_1s_bars(macro_bars, window=300)
    del macro_bars
    macro_feats_norm = macro_hmm.normalize(macro_feats)
    macro_posteriors = macro_hmm.predict_proba_forward(macro_feats_norm)
    macro_window_states = macro_posteriors.argmax(axis=1)

    # Expand macro states to per-bar (held constant within each 300-bar window)
    n_macro_windows = len(macro_window_states)
    macro_states = np.full(n_bars, -1, dtype=np.int32)
    for i in range(n_macro_windows):
        start = i * 300
        end = min((i + 1) * 300, n_bars)
        macro_states[start:end] = macro_window_states[i]

    # Compute micro features (30-bar windows)
    micro_bars = pd.DataFrame({
        "close": bars["mid"].values,
        "volume": bars["total_vol"].values,
        "spread_ticks": bars["spread_ticks"].values,
        "ofi": (bars["bid_sz"].diff().fillna(0) - bars["ask_sz"].diff().fillna(0)).values,
    })
    micro_feats, _ = micro_features_from_1s_bars(micro_bars, window=30)
    micro_feats_norm = micro_hmm.normalize(micro_feats)
    micro_posteriors = micro_hmm.predict_proba_forward(micro_feats_norm)
    micro_window_states = micro_posteriors.argmax(axis=1)

    # Expand micro states to per-bar
    n_micro_windows = len(micro_window_states)
    micro_states = np.full(n_bars, -1, dtype=np.int32)
    for i in range(n_micro_windows):
        start = i * 30
        end = min((i + 1) * 30, n_bars)
        micro_states[start:end] = micro_window_states[i]

    click.echo(f"  Regime labels: {n_macro_windows} macro windows, {n_micro_windows} micro windows")

    # Distribution
    for ms in range(macro_hmm.n_states):
        pct = (macro_states == ms).sum() / n_bars * 100
        click.echo(f"    macro={ms}: {pct:.1f}%")

    return macro_states, micro_states


def _get_month_boundaries(bar_timestamps: np.ndarray) -> list[tuple[int, int, str]]:
    """Split bar indices into calendar months.

    Args:
        bar_timestamps: unix timestamp index of 1-sec bars.

    Returns:
        List of (start_idx, end_idx, label) tuples, one per month.
    """
    dt = pd.to_datetime(bar_timestamps, unit="s", utc=True).tz_convert("US/Eastern")
    months = dt.to_period("M")

    boundaries: list[tuple[int, int, str]] = []
    unique_months = months.unique().sort_values()
    for m in unique_months:
        mask = months == m
        indices = np.where(mask)[0]
        if len(indices) > 0:
            boundaries.append((indices[0], indices[-1] + 1, str(m)))

    return boundaries


def _adjust_session_breaks(session_breaks: np.ndarray, start: int, end: int) -> np.ndarray:
    """Adjust session breaks to be relative to a slice [start:end]."""
    mask = (session_breaks >= start) & (session_breaks < end)
    return session_breaks[mask] - start


# -- Robustness checks -------------------------------------------------

def _sensitivity_analysis(
    base_config: AdaptiveStopConfig,
    p_up: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
) -> dict:
    """Perturb each param +/- 1 step, report score change."""
    base_result = run_backtest(p_up, mid, bid, ask, session_breaks, base_config)
    base_score = _score(base_result)
    perturbations = {}

    for param, values in PARAM_RANGES.items():
        current = getattr(base_config, param)
        if current not in values:
            continue
        idx = values.index(current)
        scores = {}

        for direction, offset in [("down", -1), ("up", 1)]:
            new_idx = idx + offset
            if new_idx < 0 or new_idx >= len(values):
                continue
            cfg_dict = asdict(base_config)
            cfg_dict[param] = values[new_idx]
            cfg = AdaptiveStopConfig(**cfg_dict)
            try:
                cfg.validate()
            except ValueError:
                continue
            r = run_backtest(p_up, mid, bid, ask, session_breaks, cfg)
            scores[direction] = _score(r)

        perturbations[param] = {
            "base_score": base_score,
            **scores,
        }

    return perturbations


def _monte_carlo_test(
    trade_pnls: list[float],
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Shuffle trade P&Ls, check if actual net beats 95th percentile."""
    rng = rng or np.random.default_rng(42)
    pnls = np.array(trade_pnls)
    actual_net = pnls.sum()

    shuffled_nets = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled_nets[i] = (pnls * signs).sum()

    p95 = np.percentile(shuffled_nets, 95)
    p_value = float((shuffled_nets >= actual_net).mean())

    return {
        "actual_net_pnl": float(actual_net),
        "p95_threshold": float(p95),
        "p_value": p_value,
        "significant": bool(actual_net > p95),
    }


def _surface_flatness(all_scores: list[float], best_score: float) -> dict:
    """Count configs within 80% of best score."""
    if best_score > 0:
        threshold = best_score * 0.8
    else:
        threshold = best_score * 1.2
    n_good = sum(1 for s in all_scores if s >= threshold)
    return {
        "best_score": best_score,
        "threshold_80pct": threshold,
        "n_good_configs": n_good,
        "n_total_configs": len(all_scores),
        "pct_good": n_good / len(all_scores) * 100 if all_scores else 0,
    }


# -- CLI ---------------------------------------------------------------

@click.command()
@click.option("--year", default=2025, type=int, help="Year of data to use.")
@click.option("--models", multiple=True, default=None,
              help="Specific models to sweep. Default: all tradeable.")
@click.option("--n-samples", default=1000, type=int,
              help="Number of random config samples per fold.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--window-size", default=30, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--batch-size", default=4096, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    year: int,
    models: tuple[str, ...],
    n_samples: int,
    seed: int,
    window_size: int,
    model_dir: str | None,
    batch_size: int,
    verbose: bool,
) -> None:
    """Walk-forward adaptive stop sweep with random search."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}"
    else:
        model_dir_path = Path(model_dir)

    macro_hmm_path = MODEL_DIR / "macro_hmm_v2.pkl"
    micro_hmm_path = MODEL_DIR / "micro_hmm_v2.pkl"

    # Discover tradeable models
    if models:
        model_names = list(models)
    else:
        router_path = model_dir_path.parent / "router_w30.json"
        if router_path.exists():
            with open(router_path) as f:
                router = json.load(f)
            model_names = [entry["model_id"] for entry in router if entry.get("tradeable")]
        else:
            model_names = [p.stem for p in model_dir_path.glob("pair_*.pt")]

    # Timestamped output directory
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"{'=' * 70}")
    click.echo("WALK-FORWARD ADAPTIVE STOP SWEEP (Random Search)")
    click.echo(f"{'=' * 70}")
    click.echo(f"  Year: {year}")
    click.echo(f"  Models: {model_names}")
    click.echo(f"  Window: {window_size}")
    click.echo(f"  Device: {device}")
    click.echo(f"  Samples per fold: {n_samples}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Entry thresholds: {ENTRY_THRESHOLDS}")
    click.echo(f"  Output: {run_dir}")
    click.echo()

    # -- Load data -------------------------------------------------
    click.echo("[1/4] LOADING DATA")
    bars = _load_bars(year)
    bar_timestamps = bars.index.values

    # Session breaks
    gaps = np.diff(bar_timestamps)
    session_breaks = np.where(gaps > 60)[0] + 1
    click.echo(f"  {len(session_breaks)} session breaks")

    # Month boundaries for walk-forward
    month_bounds = _get_month_boundaries(bar_timestamps)
    click.echo(f"  Months: {[m[2] for m in month_bounds]}")

    # -- Compute features + regime labels --------------------------
    click.echo("\n[2/4] COMPUTING FEATURES & REGIME LABELS")

    raw_features = _compute_features(bars)
    features = _z_score_normalize(raw_features)
    del raw_features
    gc.collect()

    mid = bars["mid"].values.astype(np.float32)
    bid = bars["bid"].values.astype(np.float32)
    ask = bars["ask"].values.astype(np.float32)

    macro_states, micro_states = _compute_regime_labels(bars, macro_hmm_path, micro_hmm_path)
    del bars
    gc.collect()

    n_bars = len(features)
    click.echo(f"  {n_bars:,} bars, {features.shape[1]} features")

    # -- Per-model walk-forward sweep ------------------------------
    click.echo(f"\n[3/4] WALK-FORWARD SWEEP")

    all_model_results: dict[str, dict] = {}
    best_configs: dict[str, dict] = {}
    all_sweep_rows: list[dict] = []
    robustness_report: dict[str, dict] = {}

    for model_name in model_names:
        model_path = model_dir_path / f"{model_name}.pt"
        if not model_path.exists():
            click.echo(f"\n  SKIP {model_name}: model not found at {model_path}")
            continue

        # Parse macro/micro state from name
        parts = model_name.replace("pair_", "").split("_")
        if len(parts) != 2:
            click.echo(f"\n  SKIP {model_name}: can't parse regime pair")
            continue
        target_macro, target_micro = int(parts[0]), int(parts[1])

        click.echo(f"\n{'-' * 70}")
        click.echo(f"  MODEL: {model_name} (macro={target_macro}, micro={target_micro})")
        click.echo(f"{'-' * 70}")

        # Regime mask
        regime_mask = (macro_states == target_macro) & (micro_states == target_micro)
        n_regime_bars = regime_mask.sum()
        click.echo(f"  Regime bars: {n_regime_bars:,} / {n_bars:,} ({n_regime_bars / n_bars * 100:.1f}%)")

        if n_regime_bars < 1000:
            click.echo(f"  SKIP: too few regime bars")
            continue

        # Load model and run inference on ALL bars
        model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        t0 = time.time()
        p_up = rolling_inference(model, features, window_size=window_size,
                                 batch_size=batch_size, device=device)
        click.echo(f"  Inference: {time.time() - t0:.1f}s")

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Mask P(up) to NaN outside this model's regime, then compute z-scores
        p_up_gated = p_up.copy()
        p_up_gated[~regime_mask] = np.nan
        z_scores = _rolling_zscore(p_up_gated, lookback=ZSCORE_LOOKBACK)
        valid_z = z_scores[np.isfinite(z_scores)]
        if len(valid_z) > 0:
            click.echo(f"  Z-score: mean={valid_z.mean():.3f}, std={valid_z.std():.3f}, "
                       f"p5={np.percentile(valid_z, 5):.2f}, p95={np.percentile(valid_z, 95):.2f}")

        # Walk-forward
        click.echo(f"\n  Walk-forward folds:")
        fold_results = []

        for test_idx in range(2, len(month_bounds)):
            train_start = month_bounds[0][0]
            train_end = month_bounds[test_idx][0]
            test_start = month_bounds[test_idx][0]
            test_end = month_bounds[test_idx][1]
            test_label = month_bounds[test_idx][2]

            # Check minimum train size
            if (train_end - train_start) < WF_MIN_TRAIN_BARS:
                continue

            # Check test has enough regime bars
            test_regime_bars = regime_mask[test_start:test_end].sum()
            if test_regime_bars < 100:
                click.echo(f"    {test_label}: SKIP (only {test_regime_bars} regime bars in test)")
                continue

            # Train slice
            train_z = z_scores[train_start:train_end]
            train_mid = mid[train_start:train_end]
            train_bid = bid[train_start:train_end]
            train_ask = ask[train_start:train_end]
            train_sb = _adjust_session_breaks(session_breaks, train_start, train_end)

            # Random search on train fold
            fold_t0 = time.time()
            best_result, best_train_score, fold_sweep_rows = _sweep_fold(
                train_z, train_mid, train_bid, train_ask, train_sb,
                n_samples, rng,
            )
            fold_elapsed = time.time() - fold_t0
            best_cfg = best_result[0] if best_result else None
            best_entry_z = best_result[1] if best_result else 0.0

            # Tag sweep rows with model/fold info
            for row in fold_sweep_rows:
                row["model"] = model_name
                row["fold"] = test_label
                row["fold_type"] = "train"
            all_sweep_rows.extend(fold_sweep_rows)

            if best_cfg is None:
                click.echo(f"    {test_label}: SKIP (no viable config in train, {fold_elapsed:.1f}s)")
                continue

            # Evaluate best config on test (OOS)
            test_z = z_scores[test_start:test_end]
            test_mid = mid[test_start:test_end]
            test_bid = bid[test_start:test_end]
            test_ask = ask[test_start:test_end]
            test_sb = _adjust_session_breaks(session_breaks, test_start, test_end)

            oos_result = _run_zscore_backtest(
                test_z, test_mid, test_bid, test_ask, test_sb,
                entry_z=best_entry_z, cfg=best_cfg,
            )

            low_confidence = oos_result.n_trades < MIN_OOS_TRADES

            # Collect per-trade dollar P&Ls for Monte Carlo (tick pnl * tick_value - per-trade commission)
            oos_trade_pnls = [
                t.pnl_ticks * MES_TICK_VALUE - best_cfg.commission_rt_dollars
                for t in oos_result.trades
            ]

            fold_results.append({
                "test_month": test_label,
                "train_bars": train_end - train_start,
                "test_bars": test_end - test_start,
                "test_regime_bars": int(test_regime_bars),
                "best_entry_z": best_entry_z,
                "best_params": asdict(best_cfg),
                "train_score": best_train_score,
                "oos_n_trades": oos_result.n_trades,
                "oos_win_rate": oos_result.win_rate,
                "oos_avg_pnl_ticks": oos_result.avg_pnl_ticks,
                "oos_profit_factor": oos_result.profit_factor,
                "oos_net_pnl": oos_result.net_pnl_dollars,
                "oos_max_dd_ticks": oos_result.max_drawdown_ticks,
                "oos_avg_hold": oos_result.avg_hold_bars,
                "low_confidence": low_confidence,
                "oos_trade_pnls": oos_trade_pnls,
                "oos_exits": {
                    "signal": oos_result.exits_signal,
                    "hard_sl": oos_result.exits_hard_sl,
                    "trail": oos_result.exits_trail,
                    "breakeven": oos_result.exits_breakeven,
                    "tier1": oos_result.exits_tier1,
                    "tier2": oos_result.exits_tier2,
                    "tier3": oos_result.exits_tier3,
                    "velocity": oos_result.exits_velocity,
                    "max_hold": oos_result.exits_max_hold,
                    "session": oos_result.exits_session_end,
                },
            })

            # Print fold summary
            pf_str = f"{oos_result.profit_factor:.2f}" if oos_result.profit_factor < 100 else "inf"
            lc_str = " [LOW CONF]" if low_confidence else ""
            click.echo(
                f"    {test_label}: "
                f"entry={best_cfg.long_entry:.2f} "
                f"SL={best_cfg.hard_sl_ticks:.0f} "
                f"BE={best_cfg.breakeven_trigger_ticks:.0f} "
                f"T1={best_cfg.tier1_activation_ticks:.0f}/{best_cfg.tier1_trail_distance:.1f} "
                f"-> OOS {oos_result.n_trades} trades, "
                f"WR={oos_result.win_rate:.1%}, "
                f"PF={pf_str}, "
                f"Net=${oos_result.net_pnl_dollars:+,.2f}"
                f"{lc_str} ({fold_elapsed:.1f}s)"
            )

        # Aggregate OOS results
        if fold_results:
            total_oos_pnl = sum(f["oos_net_pnl"] for f in fold_results)
            total_oos_trades = sum(f["oos_n_trades"] for f in fold_results)
            avg_oos_wr = np.mean([f["oos_win_rate"] for f in fold_results if f["oos_n_trades"] > 0])

            click.echo(f"\n  AGGREGATE OOS: {total_oos_trades} trades, "
                       f"WR={avg_oos_wr:.1%}, "
                       f"Net=${total_oos_pnl:+,.2f} "
                       f"({len(fold_results)} folds)")

            all_model_results[model_name] = {
                "macro_state": target_macro,
                "micro_state": target_micro,
                "n_regime_bars": int(n_regime_bars),
                "regime_pct": n_regime_bars / n_bars * 100,
                "total_oos_pnl": total_oos_pnl,
                "total_oos_trades": total_oos_trades,
                "avg_oos_win_rate": float(avg_oos_wr),
                "n_folds": len(fold_results),
                "folds": fold_results,
            }

            # Pick best config from the last fold as the "production" config
            last_fold = fold_results[-1]
            best_configs[model_name] = last_fold["best_params"]

            # -- Robustness checks --
            click.echo(f"\n  Robustness checks for {model_name}...")

            # 1. Collect all OOS trade P&Ls across folds for Monte Carlo
            all_oos_pnls: list[float] = []
            for fr in fold_results:
                all_oos_pnls.extend(fr["oos_trade_pnls"])

            mc_result = _monte_carlo_test(all_oos_pnls, rng=rng)
            mc_sig = "YES" if mc_result["significant"] else "NO"
            click.echo(f"    Monte Carlo: p={mc_result['p_value']:.4f}, significant={mc_sig}")

            # 2. Collect all sweep scores across folds for surface flatness
            all_fold_scores: list[float] = []
            for row in all_sweep_rows:
                if row["model"] == model_name and row["fold_type"] == "train":
                    s = row["score"]
                    if s != float("-inf"):
                        all_fold_scores.append(s)

            best_overall_score = max(all_fold_scores) if all_fold_scores else 0.0
            sf_result = _surface_flatness(all_fold_scores, best_overall_score)
            click.echo(f"    Surface flatness: {sf_result['n_good_configs']}/{sf_result['n_total_configs']} "
                       f"({sf_result['pct_good']:.1f}%) within 80% of best")

            # 3. Sensitivity analysis on full data with best config
            best_cfg_obj = AdaptiveStopConfig(**last_fold["best_params"])
            sens_result = _sensitivity_analysis(
                best_cfg_obj, p_up_gated, mid, bid, ask, session_breaks,
            )
            # Count fragile params (>20% score change from 1-step perturbation)
            n_fragile = 0
            for param, pdata in sens_result.items():
                base = pdata["base_score"]
                if base == 0:
                    continue
                for d in ["down", "up"]:
                    if d in pdata:
                        change_pct = abs(pdata[d] - base) / abs(base) * 100
                        if change_pct > 20:
                            n_fragile += 1
                            break
            click.echo(f"    Sensitivity: {n_fragile}/{len(sens_result)} params fragile (>20% score change)")

            robustness_report[model_name] = {
                "sensitivity": sens_result,
                "surface_flatness": sf_result,
                "monte_carlo": mc_result,
            }
        else:
            click.echo(f"\n  NO VALID FOLDS for {model_name}")
            all_model_results[model_name] = {
                "macro_state": target_macro,
                "micro_state": target_micro,
                "n_regime_bars": int(n_regime_bars),
                "regime_pct": n_regime_bars / n_bars * 100,
                "total_oos_pnl": 0.0,
                "total_oos_trades": 0,
                "n_folds": 0,
                "folds": [],
            }

        del p_up, p_up_gated
        gc.collect()

    # -- Summary ---------------------------------------------------
    click.echo(f"\n{'=' * 70}")
    click.echo("WALK-FORWARD SUMMARY")
    click.echo(f"{'=' * 70}")
    click.echo(f"\n  {'Model':<12} {'Regime%':>8} {'Folds':>6} {'OOS Trades':>11} "
               f"{'OOS WR':>7} {'OOS Net$':>10} {'Verdict':>10}")
    click.echo(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*11} {'-'*7} {'-'*10} {'-'*10}")

    for name, r in sorted(all_model_results.items(), key=lambda x: x[1]["total_oos_pnl"], reverse=True):
        verdict = "PROFITABLE" if r["total_oos_pnl"] > 0 else "UNPROFITABLE"
        wr_str = f"{r.get('avg_oos_win_rate', 0):.1%}" if r["total_oos_trades"] > 0 else "N/A"
        click.echo(
            f"  {name:<12} {r['regime_pct']:>7.1f}% {r['n_folds']:>6} "
            f"{r['total_oos_trades']:>11} {wr_str:>7} "
            f"${r['total_oos_pnl']:>+9,.2f} {verdict:>10}"
        )

    # -- Save results ----------------------------------------------
    click.echo(f"\n[4/4] SAVING RESULTS")

    # sweep_results.json (strip oos_trade_pnls to avoid bloat)
    for model_data in all_model_results.values():
        for fold in model_data.get("folds", []):
            fold.pop("oos_trade_pnls", None)

    out_path = run_dir / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(all_model_results, f, indent=2, default=str)
    click.echo(f"  {out_path}")

    # best_configs.json
    cfg_path = run_dir / "best_configs.json"
    with open(cfg_path, "w") as f:
        json.dump(best_configs, f, indent=2, default=str)
    click.echo(f"  {cfg_path}")

    # sweep_log.csv
    if all_sweep_rows:
        log_df = pd.DataFrame(all_sweep_rows)
        csv_path = run_dir / "sweep_log.csv"
        log_df.to_csv(csv_path, index=False)
        click.echo(f"  {csv_path} ({len(log_df):,} rows)")

    # robustness_report.json
    if robustness_report:
        rob_path = run_dir / "robustness_report.json"
        with open(rob_path, "w") as f:
            json.dump(robustness_report, f, indent=2, default=float)
        click.echo(f"  {rob_path}")

    elapsed = time.time() - overall_t0
    click.echo(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
