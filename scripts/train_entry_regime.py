"""Train regime-specific CNN-LSTM entry models.

Trains 3 separate models — one per HMM regime (trending, mean-reverting, choppy).
Each model only sees windows from its regime, learning regime-specific patterns.

Usage:
    python scripts/train_entry_regime.py
    python scripts/train_entry_regime.py --epochs 30 --regime trending
    python scripts/train_entry_regime.py --precompute-only
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
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataset import (
    ALL_FEATURE_NAMES,
    REGIME_FEATURES,
    EntryDataset,
    load_windows_mmap,
    precompute_windows,
    save_windows_mmap,
)
from src.models.entry_model import EntryModel
from src.regime.hmm import (
    CHOPPY,
    MEAN_REVERTING,
    TRENDING,
    STATE_NAMES,
    MarketRegimeHMM,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CACHE_DIR = DATA_DIR / "features" / "entry_windows"
HMM_FEATURES_DIR = DATA_DIR / "features" / "hmm"

HMM_FEAT_NAMES = ["return_autocorr", "hurst", "variance_ratio", "efficiency_ratio"]
HMM_WINDOW = 60  # 60 1-sec bars per HMM feature window


def _mmap_exists(name: str) -> bool:
    return (CACHE_DIR / f"{name}_features.npy").exists()


def _load_or_compute_windows(
    name: str,
    l1_path: Path,
    window_size: int,
    horizon_sec: int,
    stride: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _mmap_exists(name) and not force:
        click.echo(f"  Loading mmap cached {name}")
        return load_windows_mmap(CACHE_DIR, name)

    click.echo(f"  Computing {name} from {l1_path}...")
    t0 = time.time()
    feats, dirs, mags = precompute_windows(
        l1_path, window_size=window_size, horizon_sec=horizon_sec, stride=stride,
        output_dir=CACHE_DIR, output_name=name,
    )
    elapsed = time.time() - t0
    click.echo(f"  {len(feats):,} windows in {elapsed:.1f}s")
    return feats, dirs, mags


def _compute_regime_labels(year: int, n_entry_windows: int, stride: int, window_size: int) -> np.ndarray:
    """Compute per-window regime labels using the macro HMM.

    Returns array of shape [n_entry_windows] with values TRENDING/MEAN_REVERTING/CHOPPY.
    """
    hmm = MarketRegimeHMM()
    hmm.load(MODEL_DIR / "hmm_regime.pkl")

    feat_path = HMM_FEATURES_DIR / f"year={year}" / "data.parquet"
    if not feat_path.exists():
        click.echo(f"  [WARN] No HMM features for {year}, defaulting to CHOPPY")
        return np.full(n_entry_windows, CHOPPY, dtype=np.int32)

    t = pq.read_table(feat_path)
    feats = np.column_stack([t.column(c).to_numpy() for c in HMM_FEAT_NAMES])

    # Normalize (use data stats — same approach as train_hmm.py)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-12] = 1.0
    feats_norm = (feats - mean) / std

    # Predict regime for each 1-minute window
    regime_1m = hmm.predict(feats_norm)

    # Expand to 1-second resolution (each minute covers HMM_WINDOW seconds)
    regime_1s = np.repeat(regime_1m, HMM_WINDOW)

    # Map entry window index to the second of its last bar
    window_end_seconds = window_size + np.arange(n_entry_windows) * stride + (window_size - 1)

    # Clip to available regime labels
    valid_end = min(len(regime_1s), window_end_seconds[-1] + 1) if len(window_end_seconds) > 0 else 0
    result = np.full(n_entry_windows, CHOPPY, dtype=np.int32)
    in_range = window_end_seconds < valid_end
    result[in_range] = regime_1s[window_end_seconds[in_range]]

    return result


def _split_by_fraction(
    features: np.ndarray,
    directions: np.ndarray,
    magnitudes: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    n = len(features)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        (features[:train_end], directions[:train_end], magnitudes[:train_end]),
        (features[train_end:val_end], directions[train_end:val_end], magnitudes[train_end:val_end]),
        (features[val_end:], directions[val_end:], magnitudes[val_end:]),
    )


def train_epoch(
    model: EntryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_dir_loss = 0.0
    total_mag_loss = 0.0
    n_batches = 0

    huber = nn.HuberLoss(delta=5.0)
    mag_clip = 20.0

    for features, directions, magnitudes in loader:
        features = features.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device).clamp(-mag_clip, mag_clip)

        if label_smoothing > 0:
            smooth_dir = directions * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            smooth_dir = directions

        dir_logits, mag_pred = model(features)
        dir_logits = dir_logits.squeeze(-1)
        mag_pred = mag_pred.squeeze(-1)

        # Per-batch pos_weight for class imbalance (clamped for stability)
        pos_weight = ((1 - directions).sum() / directions.sum().clamp(min=1)).clamp(max=10.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))
        dir_loss = bce(dir_logits, smooth_dir)
        mag_loss = huber(mag_pred, magnitudes)
        loss = dir_loss + 0.1 * mag_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dir_loss += dir_loss.item()
        total_mag_loss += mag_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dir_loss": total_dir_loss / max(n_batches, 1),
        "mag_loss": total_mag_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: EntryModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    huber = nn.HuberLoss(delta=5.0)
    mag_clip = 20.0

    total_loss = 0.0
    total_dir_loss = 0.0
    total_mag_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for features, directions, magnitudes in loader:
        features = features.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device).clamp(-mag_clip, mag_clip)

        dir_logits, mag_pred = model(features)
        dir_logits = dir_logits.squeeze(-1)
        mag_pred = mag_pred.squeeze(-1)

        bce = nn.BCEWithLogitsLoss()
        dir_loss = bce(dir_logits, directions)
        mag_loss = huber(mag_pred, magnitudes)
        loss = dir_loss + 0.1 * mag_loss

        total_loss += loss.item()
        total_dir_loss += dir_loss.item()
        total_mag_loss += mag_loss.item()

        predicted = (torch.sigmoid(dir_logits) > 0.5).float()
        correct += (predicted == directions).sum().item()
        total += len(directions)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dir_loss": total_dir_loss / max(n_batches, 1),
        "mag_loss": total_mag_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
    }


def _train_one_regime(
    regime_name: str,
    train_f: np.ndarray,
    train_d: np.ndarray,
    train_m: np.ndarray,
    val_f: np.ndarray,
    val_d: np.ndarray,
    val_m: np.ndarray,
    test_f: np.ndarray,
    test_d: np.ndarray,
    test_m: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    window_size: int,
    feature_indices: list[int] | None = None,
) -> dict[str, float]:
    """Train and evaluate a single regime-specific model."""
    # Log which features this regime uses
    if feature_indices is not None:
        feat_names = [ALL_FEATURE_NAMES[i] for i in feature_indices]
        n_feat = len(feature_indices)
        click.echo(f"\n  Features ({n_feat}): {', '.join(feat_names)}")
    else:
        n_feat = train_f.shape[2]

    click.echo(f"  Train: {len(train_f):,}  Val: {len(val_f):,}  Test: {len(test_f):,}")
    click.echo(f"  Dir balance — train: {train_d.mean():.3f} long  val: {val_d.mean():.3f}  test: {test_d.mean():.3f}")

    train_ds = EntryDataset(train_f, train_d, train_m, feature_indices=feature_indices)
    val_ds = EntryDataset(val_f, val_d, val_m, feature_indices=feature_indices)
    test_ds = EntryDataset(test_f, test_d, test_m, feature_indices=feature_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = EntryModel(n_features=n_feat, seq_len=window_size).to(device)
    click.echo(f"  Model params: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    output_path = MODEL_DIR / f"entry_{regime_name}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_path)
            improved = " *"

        click.echo(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s{improved}"
        )

        if epoch - best_epoch >= 10:
            click.echo(f"  Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(output_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    click.echo(f"\n  TEST: loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.3f}")
    click.echo(f"  Model saved: {output_path}")

    return test_metrics


@click.command()
@click.option("--epochs", default=30, type=int)
@click.option("--batch-size", default=256, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--horizon", default=5, type=int)
@click.option("--stride", default=1, type=int)
@click.option("--window-size", default=30, type=int)
@click.option("--regime", default=None, type=click.Choice(["trending", "mean_reverting", "choppy"]),
              help="Train only one regime (default: all 3).")
@click.option("--precompute-only", is_flag=True)
@click.option("--force-recompute", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    stride: int,
    window_size: int,
    regime: str | None,
    precompute_only: bool,
    force_recompute: bool,
    verbose: bool,
) -> None:
    """Train regime-specific CNN-LSTM entry models."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Device: {device}")

    # ── Step 1: Load/compute feature windows per year ─────────────
    click.echo(f"\n{'='*60}")
    click.echo("[1/4] PRECOMPUTING FEATURE WINDOWS")
    click.echo(f"      window={window_size}s, horizon={horizon}s, stride={stride}s")
    click.echo(f"{'='*60}")

    year_windows = {}  # year -> (feats, dirs, mags)
    year_regimes = {}  # year -> regime_labels

    for year in [2025, 2026]:
        l1_path = DATA_DIR / "l1" / f"year={year}" / "data.parquet"
        if not l1_path.exists():
            click.echo(f"  [SKIP] No L1 data for {year}")
            continue

        name = f"year{year}_w{window_size}_h{horizon}_s{stride}"
        feats, dirs, mags = _load_or_compute_windows(
            name, l1_path, window_size, horizon, stride, force=force_recompute,
        )
        year_windows[year] = (feats, dirs, mags)

        # Compute regime labels for this year
        click.echo(f"  Computing regime labels for {year}...")
        reg_labels = _compute_regime_labels(year, len(feats), stride, window_size)
        year_regimes[year] = reg_labels

        for rid, rname in STATE_NAMES.items():
            pct = (reg_labels == rid).mean() * 100
            click.echo(f"    {rname}: {(reg_labels == rid).sum():,} ({pct:.1f}%)")

        gc.collect()

    if precompute_only:
        click.echo("\n  --precompute-only: stopping here.")
        return

    # ── Step 2: Concatenate years, split, then filter by regime ───
    click.echo(f"\n{'='*60}")
    click.echo("[2/4] COMBINING YEARS + REGIME LABELS")
    click.echo(f"{'='*60}")

    # Concatenate all years
    all_feats = np.concatenate([year_windows[y][0] for y in sorted(year_windows)])
    all_dirs = np.concatenate([np.array(year_windows[y][1]) for y in sorted(year_windows)])
    all_mags = np.concatenate([np.array(year_windows[y][2]) for y in sorted(year_windows)])
    all_regimes = np.concatenate([year_regimes[y] for y in sorted(year_regimes)])

    click.echo(f"  Total: {len(all_feats):,} windows")

    # Walk-forward split FIRST, then filter by regime within each split
    # This prevents regime-filtering from breaking chronological order
    n = len(all_feats)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    splits = {
        "train": (slice(0, train_end), "train"),
        "val": (slice(train_end, val_end), "val"),
        "test": (slice(val_end, n), "test"),
    }

    click.echo(f"  Train: {train_end:,}  Val: {val_end - train_end:,}  Test: {n - val_end:,}")

    # ── Step 3: Train per regime ──────────────────────────────────
    regime_map = {"trending": TRENDING, "mean_reverting": MEAN_REVERTING, "choppy": CHOPPY}
    regimes_to_train = [regime] if regime else ["trending", "mean_reverting", "choppy"]

    results = {}

    for rname in regimes_to_train:
        rid = regime_map[rname]
        click.echo(f"\n{'='*60}")
        click.echo(f"[3/4] TRAINING: {rname.upper()}")
        click.echo(f"{'='*60}")

        # Filter each split by regime
        train_mask = all_regimes[:train_end] == rid
        val_mask = all_regimes[train_end:val_end] == rid
        test_mask = all_regimes[val_end:] == rid

        # Need to materialize the filtered arrays (can't slice mmap with bool mask)
        train_f = np.array(all_feats[:train_end][train_mask])
        train_d = all_dirs[:train_end][train_mask]
        train_m = all_mags[:train_end][train_mask]

        val_f = np.array(all_feats[train_end:val_end][val_mask])
        val_d = all_dirs[train_end:val_end][val_mask]
        val_m = all_mags[train_end:val_end][val_mask]

        test_f = np.array(all_feats[val_end:][test_mask])
        test_d = all_dirs[val_end:][test_mask]
        test_m = all_mags[val_end:][test_mask]

        if len(train_f) < 1000 or len(val_f) < 100:
            click.echo(f"  [SKIP] Not enough data for {rname} (train={len(train_f)}, val={len(val_f)})")
            continue

        # Select regime-specific feature columns
        feat_idx = REGIME_FEATURES.get(rname, list(range(all_feats.shape[2])))

        test_metrics = _train_one_regime(
            rname, train_f, train_d, train_m,
            val_f, val_d, val_m,
            test_f, test_d, test_m,
            device, epochs, batch_size, lr, window_size,
            feature_indices=feat_idx,
        )
        results[rname] = test_metrics

        # Free filtered arrays
        del train_f, train_d, train_m, val_f, val_d, val_m, test_f, test_d, test_m
        gc.collect()

    # ── Step 4: Summary ───────────────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("SUMMARY")
    click.echo(f"{'='*60}")
    for rname, metrics in results.items():
        click.echo(f"  {rname:<16} test_acc={metrics['accuracy']:.3f}  test_loss={metrics['loss']:.4f}")

    total_elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
