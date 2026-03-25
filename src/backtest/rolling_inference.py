"""Rolling inference engine for CNN-LSTM entry models.

Runs the model on a sliding window every second, producing a continuous
P(up) signal. Batches windows for GPU efficiency.

Input: precomputed normalized feature bars [n_bars, n_features]
Output: P(up) array [n_bars] (NaN for first window_size-1 bars)
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from src.models.entry_model import EntryModel

logger = logging.getLogger(__name__)


def rolling_inference(
    model: EntryModel,
    features: np.ndarray,
    window_size: int = 30,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run model on every sliding window, return P(up) for each bar.

    Args:
        model: trained EntryModel (already on device, eval mode).
        features: [n_bars, n_features] normalized feature array.
        window_size: number of bars per window (default 30).
        batch_size: windows per GPU batch (default 4096).
        device: torch device. Inferred from model if None.

    Returns:
        p_up: [n_bars] float32 array. First (window_size - 1) entries are NaN.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_bars, n_features = features.shape
    n_windows = n_bars - window_size + 1

    if n_windows <= 0:
        return np.full(n_bars, np.nan, dtype=np.float32)

    p_up = np.full(n_bars, np.nan, dtype=np.float32)

    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            b = batch_end - batch_start

            # Build batch of windows: [b, window_size, n_features]
            # Each window i starts at bar (batch_start + i) and ends at bar (batch_start + i + window_size - 1)
            indices = np.arange(batch_start, batch_end)[:, np.newaxis] + np.arange(window_size)[np.newaxis, :]
            batch = torch.from_numpy(features[indices]).to(device)

            dir_logits, _ = model(batch)
            probs = torch.sigmoid(dir_logits.squeeze(-1)).cpu().numpy()

            # P(up) is assigned to the last bar of each window
            out_indices = indices[:, -1]
            p_up[out_indices] = probs

            if batch_start % (batch_size * 10) == 0 and batch_start > 0:
                logger.info("  Rolling inference: %d / %d windows", batch_start, n_windows)

    logger.info("  Rolling inference complete: %d windows, %d P(up) values",
                n_windows, np.isfinite(p_up).sum())

    return p_up
