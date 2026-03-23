"""Train CNN-LSTM entry model with walk-forward validation.

Walk-forward splits:
    Train:  2025 Jan-Jun
    Val:    2025 Jul-Sep
    Test:   2025 Oct - 2026 Mar

Usage:
    python scripts/train_entry.py
    python scripts/train_entry.py --epochs 50 --batch-size 256
    python scripts/train_entry.py --precompute-only
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataset import (
    EntryDataset,
    load_windows_mmap,
    precompute_windows,
    save_windows_mmap,
)
from src.models.entry_model import EntryModel

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CACHE_DIR = DATA_DIR / "features" / "entry_windows"

MES_TICK = 0.25


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
    """Load cached mmap windows or compute from L1 data.

    Returns memory-mapped arrays when loading from cache (near-zero RAM).
    """
    if _mmap_exists(name) and not force:
        click.echo(f"  Loading mmap cached {name}")
        return load_windows_mmap(CACHE_DIR, name)

    # Also check for legacy .npz and migrate
    legacy = CACHE_DIR / f"{name}.npz"
    if legacy.exists() and not force:
        click.echo(f"  Migrating legacy cache {legacy} -> mmap .npy")
        data = np.load(legacy)
        save_windows_mmap(CACHE_DIR, name, data["features"], data["directions"], data["magnitudes"])
        del data
        gc.collect()
        return load_windows_mmap(CACHE_DIR, name)

    click.echo(f"  Computing {name} from {l1_path}...")
    t0 = time.time()
    feats, dirs, mags = precompute_windows(
        l1_path, window_size=window_size, horizon_sec=horizon_sec, stride=stride,
        output_dir=CACHE_DIR, output_name=name,
    )
    elapsed = time.time() - t0
    click.echo(f"  {len(feats):,} windows in {elapsed:.1f}s")
    click.echo(f"  Cached as mmap .npy to {CACHE_DIR}")

    return feats, dirs, mags


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
    """Split chronologically into train/val/test."""
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

    huber = nn.HuberLoss(delta=5.0)  # robust to outlier magnitudes
    mag_clip = 20.0  # clip extreme moves (news spikes, flash crashes)

    for features, directions, magnitudes in loader:
        features = features.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device).clamp(-mag_clip, mag_clip)

        # Label smoothing on direction
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
        loss = dir_loss + 0.1 * mag_loss  # weight magnitude less

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
    bce = nn.BCEWithLogitsLoss()
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

        dir_loss = bce(dir_logits, directions)
        mag_loss = huber(mag_pred, magnitudes)
        loss = dir_loss + 0.1 * mag_loss

        total_loss += loss.item()
        total_dir_loss += dir_loss.item()
        total_mag_loss += mag_loss.item()

        # Direction accuracy (apply sigmoid for threshold)
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


@click.command()
@click.option("--epochs", default=30, type=int)
@click.option("--batch-size", default=256, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--horizon", default=5, type=int, help="Look-ahead seconds for labels.")
@click.option("--stride", default=5, type=int, help="Seconds between windows.")
@click.option("--window-size", default=30, type=int, help="Timesteps per window.")
@click.option("--precompute-only", is_flag=True, help="Only precompute windows, don't train.")
@click.option("--force-recompute", is_flag=True, help="Recompute windows even if cached.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    stride: int,
    window_size: int,
    precompute_only: bool,
    force_recompute: bool,
    verbose: bool,
) -> None:
    """Train CNN-LSTM entry model."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Device: {device}")

    # ── Step 1: Load/compute feature windows ──────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[1/3] PRECOMPUTING FEATURE WINDOWS")
    click.echo(f"      window={window_size}s, horizon={horizon}s, stride={stride}s")
    click.echo(f"{'='*60}")

    # Process each year's L1 data — collect mmap arrays (near-zero RAM)
    year_data = []

    for year in [2025, 2026]:
        l1_path = DATA_DIR / "l1" / f"year={year}" / "data.parquet"
        if not l1_path.exists():
            click.echo(f"  [SKIP] No L1 data for {year}")
            continue

        name = f"year{year}_w{window_size}_h{horizon}_s{stride}"
        feats, dirs, mags = _load_or_compute_windows(
            name, l1_path, window_size, horizon, stride, force=force_recompute,
        )
        year_data.append((feats, dirs, mags))
        gc.collect()

    # Concatenate into single mmap-backed file for clean split indexing
    # Write incrementally to avoid materializing all years in RAM at once
    combined_name = f"combined_w{window_size}_h{horizon}_s{stride}"
    if not _mmap_exists(combined_name) or force_recompute:
        click.echo("  Concatenating year data to combined mmap (incremental)...")
        total_n = sum(len(yd[0]) for yd in year_data)
        n_features = year_data[0][0].shape[2]

        # Create combined mmap files
        feat_mmap = np.lib.format.open_memmap(
            str(CACHE_DIR / f"{combined_name}_features.npy"), mode="w+",
            dtype=np.float32, shape=(total_n, window_size, n_features),
        )
        dir_all = np.empty(total_n, dtype=np.float32)
        mag_all = np.empty(total_n, dtype=np.float32)

        offset = 0
        for feats, dirs, mags in year_data:
            n = len(feats)
            # Copy in chunks to limit RAM (read from source mmap, write to dest mmap)
            chunk = 1000
            for i in range(0, n, chunk):
                end = min(i + chunk, n)
                feat_mmap[offset + i : offset + end] = feats[i:end]
            dir_all[offset : offset + n] = dirs[:]
            mag_all[offset : offset + n] = mags[:]
            offset += n

        feat_mmap.flush()
        del feat_mmap
        np.save(CACHE_DIR / f"{combined_name}_directions.npy", dir_all)
        np.save(CACHE_DIR / f"{combined_name}_magnitudes.npy", mag_all)
        del dir_all, mag_all
        gc.collect()

    # Free per-year refs
    del year_data
    gc.collect()

    # Load combined as mmap (near-zero RAM)
    features, directions, magnitudes = load_windows_mmap(CACHE_DIR, combined_name)

    click.echo(f"\n  Total: {len(features):,} labeled windows")
    click.echo(f"  Direction balance: {directions.mean():.3f} long / {1-directions.mean():.3f} short")
    click.echo(f"  Magnitude stats: mean={magnitudes.mean():.2f} std={magnitudes.std():.2f} ticks")

    if precompute_only:
        click.echo("\n  --precompute-only: stopping here.")
        return

    # ── Step 2: Walk-forward split ────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[2/3] WALK-FORWARD SPLIT (60/20/20)")
    click.echo(f"{'='*60}")

    # Split by index slicing — mmap slices are still mmap (no copy)
    (train_f, train_d, train_m), (val_f, val_d, val_m), (test_f, test_d, test_m) = \
        _split_by_fraction(features, directions, magnitudes)

    click.echo(f"  Train: {len(train_f):,} windows")
    click.echo(f"  Val:   {len(val_f):,} windows")
    click.echo(f"  Test:  {len(test_f):,} windows")

    train_ds = EntryDataset(train_f, train_d, train_m)
    val_ds = EntryDataset(val_f, val_d, val_m)
    test_ds = EntryDataset(test_f, test_d, test_m)

    # num_workers=0: workers fork the process, each duplicating base RAM.
    # With mmap-backed data, the OS page cache handles prefetching anyway.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # ── Step 3: Train ─────────────────────────────────────────────
    click.echo(f"\n{'='*60}")
    click.echo("[3/3] TRAINING")
    click.echo(f"{'='*60}")

    n_features = train_f.shape[2]
    model = EntryModel(
        n_features=n_features, seq_len=window_size,
    ).to(device)
    click.echo(f"  Model params: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    output_path = MODEL_DIR / "entry_model.pt"
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

        # Early stopping
        if epoch - best_epoch >= 10:
            click.echo(f"  Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load best model and evaluate on test set
    click.echo(f"\n  Loading best model (epoch {best_epoch})...")
    model.load_state_dict(torch.load(output_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    click.echo(f"\n{'='*60}")
    click.echo("TEST SET RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"  Loss:      {test_metrics['loss']:.4f}")
    click.echo(f"  Dir loss:  {test_metrics['dir_loss']:.4f}")
    click.echo(f"  Mag loss:  {test_metrics['mag_loss']:.4f}")
    click.echo(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    click.echo(f"  Model:     {output_path}")

    total_elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
