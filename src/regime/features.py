"""HMM feature computation for regime detection.

Computes 4 regime-structure features per window of 1-minute returns:
    - return_autocorr: lag-1 autocorrelation
    - hurst: Hurst exponent via R/S method (>0.5 trending, <0.5 mean-reverting)
    - variance_ratio: VR(5) — >1 trending, <1 mean-reverting
    - efficiency_ratio: |cumret|/sum(|ret|) — 1=clean trend, 0=noise
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["return_autocorr", "hurst", "variance_ratio", "efficiency_ratio"]
N_FEATURES = len(FEATURE_NAMES)


def _autocorr_1d(r: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1D array."""
    if len(r) < 3:
        return 0.0
    r0 = r[:-1]
    r1 = r[1:]
    d0 = r0 - r0.mean()
    d1 = r1 - r1.mean()
    denom = np.sqrt((d0**2).sum() * (d1**2).sum())
    if denom < 1e-15:
        return 0.0
    return float((d0 * d1).sum() / denom)


def _hurst_1d(r: np.ndarray) -> float:
    """Hurst exponent via R/S method on a 1D array."""
    if len(r) < 16:
        return 0.5

    log_rs_list = []
    log_n_list = []

    for chunk_size in [8, 16, 32]:
        if chunk_size > len(r):
            break
        num_chunks = len(r) // chunk_size
        # Vectorized: reshape into [num_chunks, chunk_size]
        trimmed = r[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
        stds = trimmed.std(axis=1)
        devs = trimmed - trimmed.mean(axis=1, keepdims=True)
        cumdevs = np.cumsum(devs, axis=1)
        rs_range = cumdevs.max(axis=1) - cumdevs.min(axis=1)
        valid = stds > 1e-15
        if valid.any():
            avg_rs = (rs_range[valid] / stds[valid]).mean()
            if avg_rs > 0:
                log_rs_list.append(np.log(avg_rs))
                log_n_list.append(np.log(chunk_size))

    if len(log_rs_list) < 2:
        return 0.5

    x = np.array(log_n_list)
    y = np.array(log_rs_list)
    n = len(x)
    denom = n * (x * x).sum() - x.sum() ** 2
    if abs(denom) < 1e-15:
        return 0.5
    h = (n * (x * y).sum() - x.sum() * y.sum()) / denom
    return max(0.0, min(1.0, float(h)))


def _variance_ratio_1d(r: np.ndarray, k: int = 5) -> float:
    """Variance ratio VR(k) on a 1D array."""
    if len(r) < k * 2:
        return 1.0
    var_1 = np.var(r, ddof=1)
    if var_1 < 1e-15:
        return 1.0
    k_ret = np.convolve(r, np.ones(k), mode="valid")
    var_k = np.var(k_ret, ddof=1)
    return float(var_k / (k * var_1))


def _efficiency_ratio_1d(r: np.ndarray) -> float:
    """Efficiency ratio on a 1D array."""
    if len(r) < 2:
        return 0.0
    total_path = np.abs(r).sum()
    if total_path < 1e-15:
        return 0.0
    return float(abs(r.sum()) / total_path)


def _batch_autocorr(chunks: np.ndarray) -> np.ndarray:
    """Vectorized lag-1 autocorrelation across all windows. [n_windows, window] -> [n_windows]."""
    r0 = chunks[:, :-1]
    r1 = chunks[:, 1:]
    d0 = r0 - r0.mean(axis=1, keepdims=True)
    d1 = r1 - r1.mean(axis=1, keepdims=True)
    denom = np.sqrt((d0**2).sum(axis=1) * (d1**2).sum(axis=1))
    num = (d0 * d1).sum(axis=1)
    return np.where(denom > 1e-15, num / denom, 0.0)


def _batch_variance_ratio(chunks: np.ndarray, k: int = 5) -> np.ndarray:
    """Vectorized variance ratio VR(k) across all windows. [n_windows, window] -> [n_windows]."""
    var_1 = chunks.var(axis=1, ddof=1)
    # k-period returns via rolling sum (convolve per row)
    n_windows, w = chunks.shape
    if w < k * 2:
        return np.ones(n_windows)
    # Use cumsum trick for rolling sum of width k
    cs = np.cumsum(chunks, axis=1)
    k_ret = cs[:, k:] - np.concatenate([np.zeros((n_windows, 1)), cs[:, :-k]], axis=1)[:, 1:]
    # Simpler: just use convolve approach but vectorized
    # Actually let's use stride_tricks
    k_ret = np.lib.stride_tricks.sliding_window_view(chunks, k, axis=1).sum(axis=2)
    var_k = k_ret.var(axis=1, ddof=1)
    return np.where(var_1 > 1e-15, var_k / (k * var_1), 1.0)


def _batch_efficiency_ratio(chunks: np.ndarray) -> np.ndarray:
    """Vectorized efficiency ratio across all windows. [n_windows, window] -> [n_windows]."""
    total_path = np.abs(chunks).sum(axis=1)
    net_move = np.abs(chunks.sum(axis=1))
    return np.where(total_path > 1e-15, net_move / total_path, 0.0)


def _compute_features_batch(chunks: np.ndarray) -> np.ndarray:
    """Compute all 4 features for a batch of windows. [n_windows, window] -> [n_windows, 4]."""
    n_windows = chunks.shape[0]
    result = np.empty((n_windows, N_FEATURES))

    result[:, 0] = _batch_autocorr(chunks)
    # Hurst still needs per-window (R/S regression with variable valid points)
    for i in range(n_windows):
        result[i, 1] = _hurst_1d(chunks[i])
    result[:, 2] = _batch_variance_ratio(chunks)
    result[:, 3] = _batch_efficiency_ratio(chunks)

    return result


def features_from_ohlcv_1m(
    df: pd.DataFrame,
    window: int = 60,
) -> np.ndarray:
    """Compute HMM features from 1-minute OHLCV bars.

    Groups consecutive 1-min close-to-close returns into non-overlapping
    windows and computes 4 features per window.

    Args:
        df: DataFrame with 'close' column and DatetimeIndex or 'ts_event' column.
        window: number of 1-min bars per feature window (default 60 = 1 hour).

    Returns:
        [n_windows, 4] array of features.
    """
    close = df["close"].values.astype(np.float64)
    log_ret = np.log(close[1:] / close[:-1])

    # Drop non-finite
    mask = np.isfinite(log_ret)
    log_ret = log_ret[mask]

    # Split into non-overlapping windows
    n_windows = len(log_ret) // window
    if n_windows == 0:
        return np.empty((0, N_FEATURES))

    # Trim to exact multiple of window
    trimmed = log_ret[: n_windows * window]
    chunks = trimmed.reshape(n_windows, window)

    result = _compute_features_batch(chunks)

    mask = np.all(np.isfinite(result), axis=1)
    return result[mask]


def features_from_l1(df: pd.DataFrame, window: int = 60) -> np.ndarray:
    """Compute HMM features from L1 tick data.

    Resamples ticks to 1-second mid prices, computes log returns,
    then groups into non-overlapping windows.
    Returns shape [n_windows, 4].
    """
    bid = df["bid_price"].values.astype(np.float64)
    ask = df["ask_price"].values.astype(np.float64)
    mid = (bid + ask) / 2.0
    ts = df["timestamp"].values

    seconds = ts.astype("datetime64[s]")

    sec_df = pd.DataFrame({"mid": mid, "second": seconds})
    sec_agg = sec_df.groupby("second").agg(mid=("mid", "last"))
    logger.info("  %d ticks -> %d 1-sec bars", len(mid), len(sec_agg))

    mids = sec_agg["mid"].values
    log_ret = np.log(mids[1:] / mids[:-1])

    mask = np.isfinite(log_ret)
    log_ret = log_ret[mask]

    # Group into windows
    n_windows = len(log_ret) // window
    if n_windows == 0:
        return np.empty((0, N_FEATURES))

    trimmed = log_ret[: n_windows * window]
    chunks = trimmed.reshape(n_windows, window)

    result = _compute_features_batch(chunks)

    mask = np.all(np.isfinite(result), axis=1)
    return result[mask]
