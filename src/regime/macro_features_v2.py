"""Macro regime feature computation (v2) for HMM regime detection.

Computes 7 features per 5-minute window (300 × 1-sec bars) for macro regime
classification. All features are strictly causal — computed from bars within
the window only.

Features:
    0: realized_vol      — std of log returns in window
    1: vol_of_vol        — std of rolling 60-bar realized vol within window
    2: return_autocorr   — lag-1 autocorrelation of 1-sec returns
    3: variance_ratio    — VR(5) Lo-MacKinlay test
    4: efficiency_ratio  — |cumret| / sum(|ret|), 1=clean trend, 0=noise
    5: volume_kurtosis   — kurtosis of per-bar volume (activity shape)
    6: return_skew        — skewness of 1-sec returns (directional bias)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.regime.features import (
    _batch_autocorr,
    _batch_efficiency_ratio,
    _batch_variance_ratio,
)

logger = logging.getLogger(__name__)

MACRO_FEATURE_NAMES = [
    "realized_vol",
    "vol_of_vol",
    "return_autocorr",
    "variance_ratio",
    "efficiency_ratio",
    "volume_kurtosis",
    "return_skew",
]
N_MACRO_FEATURES = len(MACRO_FEATURE_NAMES)

# 5-minute windows of 1-sec bars
DEFAULT_MACRO_WINDOW = 300


def _batch_realized_vol(chunks: np.ndarray) -> np.ndarray:
    """Std of log returns per window. [n_windows, window] -> [n_windows]."""
    return chunks.std(axis=1)


def _batch_vol_of_vol(chunks: np.ndarray, sub_window: int = 60) -> np.ndarray:
    """Std of rolling realized vol within each window.

    For each window, computes rolling variance via cumsum trick (O(n)),
    then takes the std of rolling stds. Captures vol clustering.

    [n_windows, window] -> [n_windows]
    """
    n_windows, w = chunks.shape
    if w < sub_window * 2:
        return np.zeros(n_windows)

    k = sub_window

    # Rolling mean via cumsum: O(n) instead of O(n*k)
    cs = np.cumsum(chunks, axis=1)
    cs2 = np.cumsum(chunks**2, axis=1)

    # Pad left with 0 for easy differencing
    cs = np.concatenate([np.zeros((n_windows, 1)), cs], axis=1)
    cs2 = np.concatenate([np.zeros((n_windows, 1)), cs2], axis=1)

    # Rolling sum and sum-of-squares for windows of size k
    # Result shape: [n_windows, w - k + 1]
    roll_sum = cs[:, k:] - cs[:, :-k]
    roll_sum2 = cs2[:, k:] - cs2[:, :-k]

    # Rolling variance = E[x^2] - E[x]^2
    roll_var = roll_sum2 / k - (roll_sum / k) ** 2
    roll_var = np.maximum(roll_var, 0.0)  # numerical safety
    rolling_vol = np.sqrt(roll_var)

    # Vol of vol: std across rolling vols
    return rolling_vol.std(axis=1)


def _batch_kurtosis(chunks: np.ndarray) -> np.ndarray:
    """Excess kurtosis per window. [n_windows, window] -> [n_windows].

    Pure numpy — avoids scipy's nan_policy overhead.
    """
    n = chunks.shape[1]
    mean = chunks.mean(axis=1, keepdims=True)
    diff = chunks - mean
    m2 = (diff**2).mean(axis=1)
    m4 = (diff**4).mean(axis=1)
    # Excess kurtosis = m4/m2^2 - 3
    return np.where(m2 > 1e-30, m4 / (m2**2) - 3.0, 0.0)


def _batch_skew(chunks: np.ndarray) -> np.ndarray:
    """Skewness per window. [n_windows, window] -> [n_windows].

    Pure numpy — avoids scipy's nan_policy overhead.
    """
    mean = chunks.mean(axis=1, keepdims=True)
    diff = chunks - mean
    m2 = (diff**2).mean(axis=1)
    m3 = (diff**3).mean(axis=1)
    # Skewness = m3 / m2^1.5
    return np.where(m2 > 1e-30, m3 / (m2**1.5), 0.0)


def macro_features_from_1s_bars(
    df: pd.DataFrame,
    window: int = DEFAULT_MACRO_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute macro regime features from 1-second OHLCV bars.

    Groups consecutive 1-sec bars into non-overlapping windows of `window`
    bars (default 300 = 5 minutes) and computes 7 features per window.

    Args:
        df: DataFrame with columns: close, volume, timestamp.
        window: number of 1-sec bars per window (default 300).

    Returns:
        Tuple of:
            features: [n_windows, 7] array of macro features.
            timestamps: [n_windows] array of window-end timestamps (last bar ts).
    """
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    n = len(close)

    # Log returns (1-sec)
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepend 0 so arrays align with bars
    log_ret = np.concatenate([[0.0], log_ret])

    # Split into non-overlapping windows
    n_windows = n // window
    if n_windows == 0:
        return np.empty((0, N_MACRO_FEATURES)), np.empty(0, dtype="datetime64[ns]")

    trimmed_n = n_windows * window

    # Reshape into [n_windows, window]
    ret_chunks = log_ret[:trimmed_n].reshape(n_windows, window)
    vol_chunks = volume[:trimmed_n].reshape(n_windows, window)

    result = np.empty((n_windows, N_MACRO_FEATURES))

    # 0: Realized volatility
    result[:, 0] = _batch_realized_vol(ret_chunks)

    # 1: Vol of vol (clustering)
    result[:, 1] = _batch_vol_of_vol(ret_chunks, sub_window=60)

    # 2: Return autocorrelation
    result[:, 2] = _batch_autocorr(ret_chunks)

    # 3: Variance ratio VR(5)
    result[:, 3] = _batch_variance_ratio(ret_chunks, k=5)

    # 4: Efficiency ratio
    result[:, 4] = _batch_efficiency_ratio(ret_chunks)

    # 5: Volume kurtosis
    result[:, 5] = _batch_kurtosis(vol_chunks)

    # 6: Return skewness
    result[:, 6] = _batch_skew(ret_chunks)

    # Timestamps: use last bar of each window
    if "timestamp" in df.columns:
        ts = df["timestamp"].values
        window_end_ts = ts[window - 1:trimmed_n:window]
    else:
        window_end_ts = np.arange(n_windows)

    # Drop non-finite rows
    mask = np.all(np.isfinite(result), axis=1)
    return result[mask], window_end_ts[mask]


def macro_features_from_l1(
    df: pd.DataFrame,
    window: int = DEFAULT_MACRO_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute macro features from raw L1 tick data.

    Resamples ticks to 1-second bars (mid price + volume), then computes
    macro features on non-overlapping windows.

    Args:
        df: DataFrame with columns: bid_price, ask_price, timestamp,
            and optionally size for volume.

    Returns:
        Tuple of (features [n_windows, 7], timestamps [n_windows]).
    """
    bid = df["bid_price"].values.astype(np.float64)
    ask = df["ask_price"].values.astype(np.float64)
    mid = (bid + ask) / 2.0
    ts = df["timestamp"].values

    seconds = ts.astype("datetime64[s]")

    # Build 1-sec bars from ticks
    sec_df = pd.DataFrame({
        "mid": mid,
        "second": seconds,
    })

    if "size" in df.columns:
        sec_df["volume"] = df["size"].values.astype(np.float64)
    else:
        sec_df["volume"] = 1.0

    sec_agg = sec_df.groupby("second").agg(
        close=("mid", "last"),
        volume=("volume", "sum"),
    )
    sec_agg["timestamp"] = sec_agg.index

    logger.info("  %d ticks -> %d 1-sec bars", len(mid), len(sec_agg))

    return macro_features_from_1s_bars(sec_agg, window=window)
