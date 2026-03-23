"""Micro regime feature extraction from 1-second OHLCV bars.

Computes 5 microstructure features per window for local regime detection:
    - spread: bid-ask spread in ticks (liquidity)
    - trade_rate: trades per second (market activity)
    - return_autocorr: lag-1 autocorrelation of 1-sec returns (trending vs MR)
    - realized_vol: std of 1-sec log returns (opportunity + risk)
    - ofi: absolute order flow imbalance (directional conviction)

These features answer "is the market tradeable right now?" at scalping timescales.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MICRO_FEATURE_NAMES = [
    "spread",
    "trade_rate",
    "return_autocorr",
    "realized_vol",
    "ofi",
]
N_MICRO_FEATURES = len(MICRO_FEATURE_NAMES)

MES_TICK = 0.25


def micro_features_from_1s_ohlcv(
    df: pd.DataFrame,
    window: int = 60,
) -> np.ndarray:
    """Compute micro regime features from 1-second OHLCV bars.

    Groups consecutive 1-sec bars into non-overlapping windows and computes
    5 summary features per window.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, vwap, timestamp.
        window: number of 1-sec bars per feature window (default 60 = 1 minute).

    Returns:
        [n_windows, 5] array of features.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    n = len(close)

    # Log returns
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0.0], log_ret])
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Spread proxy from high-low (in ticks)
    # For 1-second bars, high-low approximates the spread
    spread_ticks = (high - low) / MES_TICK

    # Trade rate proxy from volume
    trade_rate = volume.astype(np.float64)

    # OFI proxy: close position within high-low range, scaled by volume
    # (close - mid) / range gives directional pressure
    bar_range = high - low
    bar_mid = (high + low) / 2.0
    ofi_raw = np.where(
        bar_range > 0,
        (close - bar_mid) / bar_range * volume,
        0.0,
    )

    # Split into non-overlapping windows
    n_windows = n // window
    if n_windows == 0:
        return np.empty((0, N_MICRO_FEATURES))

    trimmed_n = n_windows * window

    # Reshape into [n_windows, window]
    spread_w = spread_ticks[:trimmed_n].reshape(n_windows, window)
    trade_rate_w = trade_rate[:trimmed_n].reshape(n_windows, window)
    log_ret_w = log_ret[:trimmed_n].reshape(n_windows, window)
    ofi_w = ofi_raw[:trimmed_n].reshape(n_windows, window)

    result = np.empty((n_windows, N_MICRO_FEATURES))

    # 0: Mean spread (ticks) — liquidity measure
    result[:, 0] = spread_w.mean(axis=1)

    # 1: Mean trade rate — activity measure
    result[:, 1] = trade_rate_w.mean(axis=1)

    # 2: Return autocorrelation — trending vs mean-reverting
    result[:, 2] = _batch_autocorr(log_ret_w)

    # 3: Realized volatility — std of log returns
    result[:, 3] = log_ret_w.std(axis=1)

    # 4: Mean absolute OFI — directional conviction
    result[:, 4] = np.abs(ofi_w).mean(axis=1)

    # Drop non-finite rows
    mask = np.all(np.isfinite(result), axis=1)
    return result[mask]


def _batch_autocorr(chunks: np.ndarray) -> np.ndarray:
    """Vectorized lag-1 autocorrelation. [n_windows, window] -> [n_windows]."""
    r0 = chunks[:, :-1]
    r1 = chunks[:, 1:]
    d0 = r0 - r0.mean(axis=1, keepdims=True)
    d1 = r1 - r1.mean(axis=1, keepdims=True)
    denom = np.sqrt((d0**2).sum(axis=1) * (d1**2).sum(axis=1))
    num = (d0 * d1).sum(axis=1)
    return np.where(denom > 1e-15, num / denom, 0.0)
