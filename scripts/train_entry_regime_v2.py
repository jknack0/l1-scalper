"""Train regime-specific CNN-LSTM entry models (v2).

Full walk-forward pipeline with dual-timeframe HMM regime gating.
Zero data leakage: HMMs trained before entry data, forward-only filtering,
chronological splits before regime filtering.

Usage:
    python scripts/train_entry_regime_v2.py
    python scripts/train_entry_regime_v2.py --fold 1
    python scripts/train_entry_regime_v2.py --window-sizes 30,60
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataset import EntryDataset, IndexedEntryDataset, load_windows_mmap, precompute_windows, precompute_windows_bracket
from src.models.entry_model import EntryModel
from src.regime.macro_features_v2 import DEFAULT_MACRO_WINDOW, MACRO_FEATURE_NAMES
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_features_v2 import DEFAULT_MICRO_WINDOW, MICRO_FEATURE_NAMES
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2
from src.regime.regime_pair import RegimePairConfig, RegimePairRouter
from src.regime.regime_validator import save_report, validate_regime_pairs

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "regime_v2"
CACHE_DIR = DATA_DIR / "features" / "entry_windows"

MACRO_FEATURES_DIR = DATA_DIR / "features" / "macro_hmm_v2"
MICRO_FEATURES_DIR = DATA_DIR / "features" / "micro_hmm_v2"
L1_DIR = DATA_DIR / "l1"

# Walk-forward fold definitions
# HMM trains on years strictly before the gap.
# Gap provides HMM forward filter warm-up.
# Entry model uses data after the gap.
FOLDS = {
    1: {
        "hmm_train_years": list(range(2011, 2024)),  # 2011-2023
        "gap_years": [2024],                          # Q1-Q2 2024 warm-up
        "entry_years": [2024, 2025],                  # Q3 2024 - Q4 2025
        "description": "HMM:2011-2023, Gap:2024-H1, Entry:2024-H2 to 2025",
    },
    2: {
        "hmm_train_years": list(range(2011, 2025)),  # 2011-2024
        "gap_years": [2025],                          # Q1 2025 warm-up
        "entry_years": [2025, 2026],                  # Q2 2025 - 2026
        "description": "HMM:2011-2024, Gap:2025-Q1, Entry:2025-Q2 to 2026",
    },
}


def _load_features(feat_dir: Path, years: list[int], feat_names: list[str]) -> np.ndarray:
    """Load precomputed HMM features for given years."""
    all_feats = []
    for year in years:
        path = feat_dir / f"year={year}" / "data.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path)
        feats = np.column_stack([t.column(c).to_numpy() for c in feat_names])
        all_feats.append(feats)
    if not all_feats:
        return np.empty((0, len(feat_names)))
    return np.concatenate(all_feats)


def _label_entry_windows(
    n_entry_windows: int,
    macro_labels: np.ndarray,
    micro_labels: np.ndarray,
    macro_window: int,
    micro_window: int,
    entry_window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign (macro, micro) labels to each entry window.

    Maps entry window index to enclosing macro/micro window index.
    Uses the regime label at the END of each entry window (causal).

    Returns (macro_labels_per_entry, micro_labels_per_entry).
    """
    # Entry window end bar index
    window_end_bars = entry_window_size + np.arange(n_entry_windows) * stride + (entry_window_size - 1)

    # Map to macro/micro window indices
    macro_indices = window_end_bars // macro_window
    micro_indices = window_end_bars // micro_window

    # Clip to available labels
    macro_valid = macro_indices < len(macro_labels)
    micro_valid = micro_indices < len(micro_labels)
    valid = macro_valid & micro_valid

    macro_out = np.full(n_entry_windows, -1, dtype=np.int32)
    micro_out = np.full(n_entry_windows, -1, dtype=np.int32)

    macro_out[valid] = macro_labels[macro_indices[valid]]
    micro_out[valid] = micro_labels[micro_indices[valid]]

    return macro_out, micro_out


def train_epoch(
    model: EntryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
) -> dict[str, float]:
    """Train one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    total_dir_loss = 0.0
    total_mag_loss = 0.0
    n_batches = 0
    huber = nn.HuberLoss(delta=5.0)

    for features, directions, magnitudes in loader:
        features = features.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device).clamp(-20, 20)

        if label_smoothing > 0:
            smooth_dir = directions * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            smooth_dir = directions

        dir_logits, mag_pred = model(features)
        dir_logits = dir_logits.squeeze(-1)
        mag_pred = mag_pred.squeeze(-1)

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
    """Evaluate model. Returns metrics dict."""
    model.eval()
    huber = nn.HuberLoss(delta=5.0)
    total_loss = 0.0
    total_dir_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for features, directions, magnitudes in loader:
        features = features.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device).clamp(-20, 20)

        dir_logits, mag_pred = model(features)
        dir_logits = dir_logits.squeeze(-1)
        mag_pred = mag_pred.squeeze(-1)

        bce = nn.BCEWithLogitsLoss()
        dir_loss = bce(dir_logits, directions)
        mag_loss = huber(mag_pred, magnitudes)
        loss = dir_loss + 0.1 * mag_loss

        total_loss += loss.item()
        total_dir_loss += dir_loss.item()
        predicted = (torch.sigmoid(dir_logits) > 0.5).float()
        correct += (predicted == directions).sum().item()
        total += len(directions)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dir_loss": total_dir_loss / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
    }


def _train_one_model(
    model_id: str,
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
    output_dir: Path,
) -> dict[str, float]:
    """Train and evaluate a single entry model."""
    n_feat = train_f.shape[2]
    click.echo(f"\n  Model: {model_id}")
    click.echo(f"  Train: {len(train_f):,}  Val: {len(val_f):,}  Test: {len(test_f):,}")
    click.echo(f"  Features: {n_feat}, Window: {window_size}")
    click.echo(f"  Dir balance — train: {train_d.mean():.3f}  val: {val_d.mean():.3f}  test: {test_d.mean():.3f}")

    train_ds = EntryDataset(train_f, train_d, train_m)
    val_ds = EntryDataset(val_f, val_d, val_m)
    test_ds = EntryDataset(test_f, test_d, test_m)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EntryModel(n_features=n_feat, seq_len=window_size).to(device)
    click.echo(f"  Params: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    output_path = output_dir / f"{model_id}.pt"
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
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.3f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s{improved}"
        )

        if epoch - best_epoch >= 7:
            click.echo(f"  Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load best and evaluate test
    model.load_state_dict(torch.load(output_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    click.echo(f"  TEST: loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.3f}")
    click.echo(f"  Saved: {output_path}")

    return test_metrics


def _train_one_model_indexed(
    model_id: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    window_size: int,
    output_dir: Path,
) -> dict[str, float]:
    """Train and evaluate using pre-built IndexedEntryDataset (zero-copy from mmap)."""
    n_feat = train_ds.features.shape[2]
    click.echo(f"\n  Model: {model_id}")
    click.echo(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    click.echo(f"  Features: {n_feat}, Window: {window_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EntryModel(n_features=n_feat, seq_len=window_size).to(device)
    click.echo(f"  Params: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    output_path = output_dir / f"{model_id}.pt"
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
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.3f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s{improved}"
        )

        if epoch - best_epoch >= 7:
            click.echo(f"  Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    model.load_state_dict(torch.load(output_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    click.echo(f"  TEST: loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.3f}")
    click.echo(f"  Saved: {output_path}")

    return test_metrics


@click.command()
@click.option("--fold", default=1, type=int, help="Walk-forward fold (1 or 2).")
@click.option("--epochs", default=30, type=int)
@click.option("--batch-size", default=256, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--horizon", default=5, type=int, help="Forward return horizon in seconds.")
@click.option("--stride", default=10, type=int, help="Window stride in bars. Default 10 (~620K windows/year vs 6.2M).")
@click.option("--window-sizes", default="30,60", help="Entry window sizes to test (comma-separated).")
@click.option("--bracket/--no-bracket", default=True, help="Use bracket exit labels (9:9 TP/SL, 5min timeout). Default: bracket.")
@click.option("--macro-hmm-path", default=None, type=str, help="Load pre-fitted macro HMM instead of refitting.")
@click.option("--micro-hmm-path", default=None, type=str, help="Load pre-fitted micro HMM instead of refitting.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    stride: int,
    window_sizes: str,
    bracket: bool,
    macro_hmm_path: str | None,
    micro_hmm_path: str | None,
    verbose: bool,
) -> None:
    """Train regime-specific entry models with zero data leakage."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fold not in FOLDS:
        click.echo(f"ERROR: fold must be one of {list(FOLDS.keys())}")
        return

    fold_cfg = FOLDS[fold]
    ws_list = [int(x) for x in window_sizes.split(",")]

    click.echo(f"Fold {fold}: {fold_cfg['description']}")
    click.echo(f"Window sizes to test: {ws_list}")
    click.echo(f"Device: {device}")
    click.echo()

    # ── Step 1: Load/fit HMMs ────────────────────────────────────
    click.echo("=" * 60)
    click.echo("[1/6] LOADING/FITTING HMMs")
    click.echo("=" * 60)

    # Macro HMM
    if macro_hmm_path:
        click.echo(f"  Loading pre-fitted macro HMM: {macro_hmm_path}")
        macro_hmm = MacroRegimeHMMv2()
        macro_hmm.load(macro_hmm_path)
        click.echo(f"  Macro: {macro_hmm.n_states} states")
    else:
        macro_train = _load_features(MACRO_FEATURES_DIR, fold_cfg["hmm_train_years"], MACRO_FEATURE_NAMES)
        if len(macro_train) == 0:
            click.echo("ERROR: No macro training features.")
            return
        click.echo(f"  Macro train: {len(macro_train):,} windows")
        macro_hmm = MacroRegimeHMMv2(n_mix=2, covariance_type="diag")
        macro_hmm.fit_with_bic(macro_train, n_states_candidates=[3, 4, 5])
        click.echo(f"  Macro: {macro_hmm.n_states} states selected")
        del macro_train
        gc.collect()

    # Micro HMM
    if micro_hmm_path:
        click.echo(f"  Loading pre-fitted micro HMM: {micro_hmm_path}")
        micro_hmm = MicroRegimeHMMv2()
        micro_hmm.load(micro_hmm_path)
        click.echo(f"  Micro: {micro_hmm.n_states} states")
    else:
        micro_train = _load_features(MICRO_FEATURES_DIR, fold_cfg["hmm_train_years"], MICRO_FEATURE_NAMES)
        if len(micro_train) == 0:
            click.echo("ERROR: No micro training features.")
            return
        click.echo(f"  Micro train: {len(micro_train):,} windows")
        micro_hmm = MicroRegimeHMMv2()
        micro_hmm.fit_with_bic(micro_train, n_states_candidates=[3, 4, 5])
        click.echo(f"  Micro: {micro_hmm.n_states} states selected")
        del micro_train
        gc.collect()

    # Save HMMs
    fold_model_dir = MODEL_DIR / f"regime_v2_fold{fold}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    macro_hmm.save(fold_model_dir / "macro_hmm.pkl")
    micro_hmm.save(fold_model_dir / "micro_hmm.pkl")

    # ── Step 2: Label entry data with forward-only filtering ──────
    click.echo(f"\n{'=' * 60}")
    click.echo("[2/6] LABELING ENTRY DATA (forward-only filtering)")
    click.echo("=" * 60)

    # Load macro/micro features for entry years (includes gap for warm-up)
    all_entry_years = fold_cfg["gap_years"] + fold_cfg["entry_years"]
    macro_entry = _load_features(MACRO_FEATURES_DIR, all_entry_years, MACRO_FEATURE_NAMES)
    micro_entry = _load_features(MICRO_FEATURES_DIR, all_entry_years, MICRO_FEATURE_NAMES)

    if len(macro_entry) == 0 or len(micro_entry) == 0:
        click.echo("ERROR: No features for entry years.")
        return

    # Normalize using HMM training stats (stored in HMM)
    macro_entry_norm = macro_hmm.normalize(macro_entry)
    micro_entry_norm = micro_hmm.normalize(micro_entry)

    # Forward-only filtering (NO Viterbi — this is the critical anti-leakage step)
    click.echo("  Forward-only filtering on macro...")
    macro_posteriors = macro_hmm.predict_proba_forward(macro_entry_norm)
    macro_labels = macro_posteriors.argmax(axis=1)

    click.echo("  Forward-only filtering on micro...")
    micro_posteriors = micro_hmm.predict_proba_forward(micro_entry_norm)
    micro_labels = micro_posteriors.argmax(axis=1)

    for s in range(macro_hmm.n_states):
        pct = (macro_labels == s).mean() * 100
        click.echo(f"  Macro state {s}: {(macro_labels == s).sum():,} ({pct:.1f}%)")
    for s in range(micro_hmm.n_states):
        pct = (micro_labels == s).mean() * 100
        click.echo(f"  Micro state {s}: {(micro_labels == s).sum():,} ({pct:.1f}%)")

    del macro_entry, micro_entry, macro_entry_norm, micro_entry_norm
    gc.collect()

    # ── Step 3: Compute entry windows per year ────────────────────
    all_results = {}

    for window_size in ws_list:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"[3/6] ENTRY WINDOWS (window_size={window_size})")
        click.echo("=" * 60)

        year_windows = {}
        for year in fold_cfg["entry_years"]:
            l1_path = L1_DIR / f"year={year}" / "data.parquet"
            if not l1_path.exists():
                click.echo(f"  [SKIP] No L1 data for {year}")
                continue

            label_tag = "bracket9" if bracket else f"h{horizon}"
            name = f"v2_fold{fold}_year{year}_w{window_size}_{label_tag}_s{stride}"
            cache_path = CACHE_DIR / f"{name}_features.npy"

            if cache_path.exists():
                click.echo(f"  Loading cached {name}...")
                feats, dirs, mags = load_windows_mmap(CACHE_DIR, name)
            else:
                click.echo(f"  Computing {name}...")
                t0 = time.time()
                if bracket:
                    feats, dirs, mags = precompute_windows_bracket(
                        l1_path, window_size=window_size,
                        tp_ticks=9.0, sl_ticks=9.0, max_hold_bars=300,
                        stride=stride,
                        output_dir=CACHE_DIR, output_name=name,
                    )
                else:
                    feats, dirs, mags = precompute_windows(
                        l1_path, window_size=window_size,
                        horizon_sec=horizon, stride=stride,
                        output_dir=CACHE_DIR, output_name=name,
                    )
                click.echo(f"  {len(feats):,} windows in {time.time() - t0:.1f}s")

            year_windows[year] = (feats, dirs, mags)

        if not year_windows:
            click.echo("  No entry data available for this fold.")
            continue

        # Concatenate
        all_feats = np.concatenate([year_windows[y][0] for y in sorted(year_windows)])
        all_dirs = np.concatenate([np.array(year_windows[y][1]) for y in sorted(year_windows)])
        all_mags = np.concatenate([np.array(year_windows[y][2]) for y in sorted(year_windows)])

        click.echo(f"  Total: {len(all_feats):,} windows")

        # Assign regime labels to entry windows
        entry_macro, entry_micro = _label_entry_windows(
            len(all_feats), macro_labels, micro_labels,
            DEFAULT_MACRO_WINDOW, DEFAULT_MICRO_WINDOW,
            window_size, stride,
        )

        # ── Step 4: Walk-forward split FIRST, then filter by regime ──
        click.echo(f"\n{'=' * 60}")
        click.echo(f"[4/6] WALK-FORWARD SPLIT + REGIME VALIDATION (w={window_size})")
        click.echo("=" * 60)

        n = len(all_feats)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        click.echo(f"  Train: {train_end:,}  Val: {val_end - train_end:,}  Test: {n - val_end:,}")

        # Validate regime pairs on training split
        # For bracket labels, magnitudes are the realized P&L in ticks (+9, -9, or timeout value)
        # Convert to price units for the validator
        MES_TICK = 0.25
        train_mags_price = all_mags[:train_end] * MES_TICK

        # For bracket labels: lower baseline threshold since 50% is breakeven with symmetric 9:9
        # Any edge above 50% is meaningful. Also skip consistency check (bracket labels are single outcome)
        min_baseline = 0.50 if bracket else 0.52

        report = validate_regime_pairs(
            macro_labels=entry_macro[:train_end],
            micro_labels=entry_micro[:train_end],
            forward_returns_5s=train_mags_price,
            forward_returns_15s=train_mags_price,  # same for bracket (single outcome, no multi-horizon)
            features=all_feats[:train_end, -1, :5].astype(np.float64),  # last bar, first 5 features
            macro_n_states=macro_hmm.n_states,
            micro_n_states=micro_hmm.n_states,
            macro_hmm_means=macro_hmm.weighted_means,
            micro_hmm_means=micro_hmm.weighted_means,
            min_baseline_acc=min_baseline,
            min_magnitude_ticks=0.0,  # bracket labels already encode the exit, no separate magnitude filter
        )

        # Save validation report
        report_dir = RESULTS_DIR / f"fold{fold}_w{window_size}"
        report_dir.mkdir(parents=True, exist_ok=True)
        save_report(report, report_dir / "validation_report.json")

        click.echo(f"  Tradeable pairs: {report.tradeable_pairs}")
        click.echo(f"  Merge map: {report.merge_map}")
        click.echo(f"  Skip pairs: {report.skip_pairs}")

        # Build router
        configs = []
        for pair in report.pairs:
            pair_key = f"{pair.macro_state}_{pair.micro_state}"
            model_id = f"pair_{pair_key}"

            if pair.classification == "trade":
                tradeable = True
                merged_into = None
            elif pair.classification == "merge" and pair_key in report.merge_map:
                tradeable = True
                merged_into = f"pair_{report.merge_map[pair_key]}"
            else:
                tradeable = False
                merged_into = None

            configs.append(RegimePairConfig(
                macro_state=pair.macro_state, micro_state=pair.micro_state,
                model_id=model_id, tradeable=tradeable, merged_into=merged_into,
                n_train_samples=pair.n_samples, baseline_accuracy=pair.baseline_accuracy,
                mean_magnitude_ticks=pair.mean_magnitude_ticks,
            ))

        router = RegimePairRouter(configs)
        router.save(fold_model_dir / f"router_w{window_size}.json")

        # ── Step 5: Train regime-specific models ──────────────────
        click.echo(f"\n{'=' * 60}")
        click.echo(f"[5/6] TRAINING REGIME-SPECIFIC MODELS (w={window_size})")
        click.echo("=" * 60)

        model_results = {}

        # Get unique model IDs to train (deduplicated via merges)
        models_to_train = router.tradeable_model_ids
        click.echo(f"  Models to train: {models_to_train}")

        for model_id in sorted(models_to_train):
            # Collect all regime pairs that map to this model
            pair_keys = []
            for cfg in router.all_configs:
                if cfg.tradeable and cfg.effective_model_id == model_id:
                    pair_keys.append((cfg.macro_state, cfg.micro_state))

            # Build masks for each split
            train_mask = np.zeros(train_end, dtype=bool)
            val_mask = np.zeros(val_end - train_end, dtype=bool)
            test_mask = np.zeros(n - val_end, dtype=bool)

            for m_s, u_s in pair_keys:
                train_mask |= (entry_macro[:train_end] == m_s) & (entry_micro[:train_end] == u_s)
                val_mask |= (entry_macro[train_end:val_end] == m_s) & (entry_micro[train_end:val_end] == u_s)
                test_mask |= (entry_macro[val_end:] == m_s) & (entry_micro[val_end:] == u_s)

            # Use integer indices into mmap — zero copy, ~0 RAM
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0] + train_end
            test_idx = np.where(test_mask)[0] + val_end

            if len(train_idx) < 200 or len(val_idx) < 50:
                click.echo(f"\n  [SKIP] {model_id}: insufficient data "
                          f"(train={len(train_idx)}, val={len(val_idx)})")
                continue

            # IndexedEntryDataset reads from mmap on-demand per batch
            train_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, train_idx)
            val_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, val_idx)
            test_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, test_idx)

            metrics = _train_one_model_indexed(
                model_id, train_ds, val_ds, test_ds,
                device, epochs, batch_size, lr, window_size,
                output_dir=fold_model_dir / f"w{window_size}_h{horizon}",
            )
            model_results[model_id] = metrics
            gc.collect()

        # Train fallback model on ALL tradeable data
        click.echo(f"\n  Training fallback model...")
        all_tradeable_mask_train = np.zeros(train_end, dtype=bool)
        all_tradeable_mask_val = np.zeros(val_end - train_end, dtype=bool)
        all_tradeable_mask_test = np.zeros(n - val_end, dtype=bool)

        for cfg in router.all_configs:
            if cfg.tradeable:
                m_s, u_s = cfg.macro_state, cfg.micro_state
                all_tradeable_mask_train |= (entry_macro[:train_end] == m_s) & (entry_micro[:train_end] == u_s)
                all_tradeable_mask_val |= (entry_macro[train_end:val_end] == m_s) & (entry_micro[train_end:val_end] == u_s)
                all_tradeable_mask_test |= (entry_macro[val_end:] == m_s) & (entry_micro[val_end:] == u_s)

        fb_train_idx = np.where(all_tradeable_mask_train)[0]
        fb_val_idx = np.where(all_tradeable_mask_val)[0] + train_end
        fb_test_idx = np.where(all_tradeable_mask_test)[0] + val_end

        if len(fb_train_idx) >= 200:
            fb_train_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, fb_train_idx)
            fb_val_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, fb_val_idx)
            fb_test_ds = IndexedEntryDataset(all_feats, all_dirs, all_mags, fb_test_idx)

            fb_metrics = _train_one_model_indexed(
                "fallback", fb_train_ds, fb_val_ds, fb_test_ds,
                device, epochs, batch_size, lr, window_size,
                output_dir=fold_model_dir / f"w{window_size}_h{horizon}",
            )
            model_results["fallback"] = fb_metrics
        gc.collect()

        all_results[f"w{window_size}"] = model_results

        # Cleanup
        del all_feats, all_dirs, all_mags, entry_macro, entry_micro
        gc.collect()

    # ── Step 6: Summary ───────────────────────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("[6/6] SUMMARY")
    click.echo("=" * 60)

    for ws_key, results in all_results.items():
        click.echo(f"\n  {ws_key}:")
        for model_id, metrics in sorted(results.items()):
            click.echo(f"    {model_id:<20} test_acc={metrics['accuracy']:.3f}  test_loss={metrics['loss']:.4f}")

    # Save results
    results_path = RESULTS_DIR / f"fold{fold}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    click.echo(f"\n  Results saved: {results_path}")

    elapsed = time.time() - overall_t0
    click.echo(f"\nDone — {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
