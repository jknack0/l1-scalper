"""Micro regime feature computation (v2) for local regime detection.

Computes 6 features per 30-second window (30 × 1-sec bars) for micro regime
classification at scalping timescales. Uses real L1 bid/ask spread instead
of the high-low proxy used in v1.

Features:
    0: spread_mean       — mean bid-ask spread in ticks (liquidity cost)
    1: spread_cv         — coefficient of variation of spread (stability)
    2: trade_rate        — mean volume per bar (market activity)
    3: ofi_momentum      — cumulative order flow imbalance, normalized
    4: return_autocorr   — lag-1 autocorrelation of 1-sec returns
    5: vol_ratio         — vol(last half) / vol(first half), vol acceleration
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.regime.features import _batch_autocorr

logger = logging.getLogger(__name__)

MICRO_FEATURE_NAMES = [
    "spread_mean",
    "spread_cv",
    "trade_rate",
    "ofi_momentum",
    "return_autocorr",
    "vol_ratio",
]
N_MICRO_FEATURES = len(MICRO_FEATURE_NAMES)

MES_TICK = 0.25
DEFAULT_MICRO_WINDOW = 30  # 30-sec windows


def micro_features_from_1s_bars(
    df: pd.DataFrame,
    window: int = DEFAULT_MICRO_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute micro regime features from 1-second bars with real spread.

    Requires columns: close, volume, spread_ticks, ofi, timestamp.
    spread_ticks and ofi should be precomputed from L1 bid/ask data.

    Args:
        df: DataFrame with 1-sec bar data including spread and OFI.
        window: number of 1-sec bars per window (default 30).

    Returns:
        Tuple of (features [n_windows, 6], timestamps [n_windows]).
    """
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    spread = df["spread_ticks"].values.astype(np.float64)
    ofi = df["ofi"].values.astype(np.float64)

    n = len(close)

    # Log returns
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0.0], log_ret])
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Split into non-overlapping windows
    n_windows = n // window
    if n_windows == 0:
        return np.empty((0, N_MICRO_FEATURES)), np.empty(0, dtype="datetime64[ns]")

    trimmed_n = n_windows * window

    # Reshape into [n_windows, window]
    spread_w = spread[:trimmed_n].reshape(n_windows, window)
    volume_w = volume[:trimmed_n].reshape(n_windows, window)
    ofi_w = ofi[:trimmed_n].reshape(n_windows, window)
    ret_w = log_ret[:trimmed_n].reshape(n_windows, window)

    result = np.empty((n_windows, N_MICRO_FEATURES))

    # 0: Mean spread in ticks
    result[:, 0] = spread_w.mean(axis=1)

    # 1: Spread coefficient of variation (stability)
    spread_mean = spread_w.mean(axis=1)
    spread_std = spread_w.std(axis=1)
    result[:, 1] = np.where(spread_mean > 1e-10, spread_std / spread_mean, 0.0)

    # 2: Mean trade rate (volume per bar)
    result[:, 2] = volume_w.mean(axis=1)

    # 3: OFI momentum — cumulative OFI over window, normalized by total volume
    cum_ofi = ofi_w.sum(axis=1)
    total_vol = volume_w.sum(axis=1)
    result[:, 3] = np.where(total_vol > 0, cum_ofi / total_vol, 0.0)

    # 4: Return autocorrelation
    result[:, 4] = _batch_autocorr(ret_w)

    # 5: Vol ratio — realized vol of second half / first half
    half = window // 2
    vol_first = ret_w[:, :half].std(axis=1)
    vol_second = ret_w[:, half:].std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[:, 5] = np.where(vol_first > 1e-15, vol_second / vol_first, 1.0)

    # Timestamps
    if "timestamp" in df.columns:
        ts = df["timestamp"].values
        window_end_ts = ts[window - 1:trimmed_n:window]
    else:
        window_end_ts = np.arange(n_windows)

    # Drop non-finite
    mask = np.all(np.isfinite(result), axis=1)
    return result[mask], window_end_ts[mask]


def micro_features_from_l1(
    df: pd.DataFrame,
    window: int = DEFAULT_MICRO_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute micro features from raw L1 tick data.

    Resamples ticks to 1-second bars with real bid-ask spread and
    order flow imbalance, then computes micro features.

    Args:
        df: DataFrame with columns: bid_price, ask_price, bid_size,
            ask_size, timestamp, and optionally price/size for trades.

    Returns:
        Tuple of (features [n_windows, 6], timestamps [n_windows]).
    """
    bid = df["bid_price"].values.astype(np.float64)
    ask = df["ask_price"].values.astype(np.float64)
    mid = (bid + ask) / 2.0
    spread_raw = (ask - bid) / MES_TICK  # in ticks
    ts = df["timestamp"].values
    seconds = ts.astype("datetime64[s]")

    # OFI: change in bid size - change in ask size (instantaneous)
    bid_sz = df["bid_size"].values.astype(np.float64)
    ask_sz = df["ask_size"].values.astype(np.float64)
    d_bid = np.diff(bid_sz, prepend=bid_sz[0])
    d_ask = np.diff(ask_sz, prepend=ask_sz[0])
    ofi_tick = d_bid - d_ask

    sec_df = pd.DataFrame({
        "mid": mid,
        "spread": spread_raw,
        "ofi": ofi_tick,
        "second": seconds,
    })

    if "size" in df.columns:
        sec_df["volume"] = df["size"].values.astype(np.float64)
    else:
        sec_df["volume"] = 1.0

    # Aggregate to 1-sec bars
    sec_agg = sec_df.groupby("second").agg(
        close=("mid", "last"),
        volume=("volume", "sum"),
        spread_ticks=("spread", "mean"),  # mean spread over the second
        ofi=("ofi", "sum"),  # cumulative OFI over the second
    )
    sec_agg["timestamp"] = sec_agg.index

    logger.info("  %d ticks -> %d 1-sec bars", len(mid), len(sec_agg))

    return micro_features_from_1s_bars(sec_agg, window=window)
