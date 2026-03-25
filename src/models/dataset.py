"""Vectorized dataset for CNN-LSTM entry model training.

Resamples L1 ticks to 1-second bars, computes 12 features vectorized,
builds labeled [window, 12] sliding windows.

Label construction:
    For each window ending at second t:
    - direction: 1 if mid(t + horizon) > mid(t), else 0
    - magnitude: (mid(t + horizon) - mid(t)) / tick_size
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MES_TICK = 0.25

# All features computed per 1-sec bar (superset across all regimes)
ALL_FEATURE_NAMES = [
    # ── Common (0-4) ──
    "log_return",           # 0: 1-sec log return
    "spread",               # 1: bid-ask spread in ticks
    "trade_count",          # 2: trades this second
    "signed_volume",        # 3: buy_vol - sell_vol (raw)
    "lee_ready",            # 4: sign(trade_price - mid)
    # ── Mean-reverting / Choppy (5-8) ──
    "microprice_displace",  # 5: microprice - mid in ticks (book lean → fade)
    "trade_imbalance",      # 6: (buy - sell) / total (exhaustion signal)
    "ofi",                  # 7: bid_diff - ask_diff (instantaneous)
    "log_return_sq",        # 8: squared return (vol proxy → reversals)
    # ── Trending (9-14) ──
    "cum_ofi_5",            # 9: cumulative OFI over 5 bars
    "cum_ofi_15",           # 10: cumulative OFI over 15 bars
    "price_velocity_5",     # 11: cumulative return over 5 bars
    "price_velocity_15",    # 12: cumulative return over 15 bars
    "volume_accel",         # 13: trade_count / rolling_mean(10) - 1
    "book_imbalance",       # 14: bid_sz / (bid_sz + ask_sz) — persistent pressure
]

# Per-regime feature selections (indices into ALL_FEATURE_NAMES)
REGIME_FEATURES = {
    "mean_reverting": [0, 1, 2, 3, 4, 5, 6, 7, 8],       # common + fade signals
    "choppy":         list(range(15)),                      # all features — let model decide
    "trending":       [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14],  # common + momentum
    "all":            list(range(len(ALL_FEATURE_NAMES))),  # everything
}

# Backward compat — default feature names for non-regime-split training
FEATURE_NAMES = ALL_FEATURE_NAMES


def _resample_to_1sec(df: pd.DataFrame) -> pd.DataFrame:
    """Resample L1 ticks to 1-second bars.

    Returns DataFrame indexed by integer second (unix timestamp) with columns:
        bid, ask, mid, spread_ticks,
        trade_count, buy_vol, sell_vol, total_vol,
        last_trade_price, max_trade_size, log_return
    """
    ts = df["timestamp"].values
    seconds = ts.astype("datetime64[s]").astype(np.int64)

    bid = df["bid_price"].values.astype(np.float32)
    ask = df["ask_price"].values.astype(np.float32)
    mid = (bid + ask) / 2.0
    bid_sz = df["bid_size"].values.astype(np.float32) if "bid_size" in df.columns else np.ones(len(df), dtype=np.float32)
    ask_sz = df["ask_size"].values.astype(np.float32) if "ask_size" in df.columns else np.ones(len(df), dtype=np.float32)
    price = df["price"].values.astype(np.float32)
    size = df["size"].values.astype(np.int32)

    # Classify trades: Lee-Ready (price vs mid)
    is_trade = (size > 0) & np.isfinite(price) & (price > 0)
    trade_side = np.where(price > mid, 1, np.where(price < mid, -1, 0))

    # Build per-tick DataFrame for groupby — reuse df to avoid doubling RAM
    df_reuse = pd.DataFrame({
        "sec": seconds,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "bid_sz": bid_sz,
        "ask_sz": ask_sz,
        "is_trade": is_trade,
        "buy_vol": np.where(is_trade & (trade_side > 0), size, 0),
        "sell_vol": np.where(is_trade & (trade_side < 0), size, 0),
        "trade_price": np.where(is_trade, price, np.nan),
        "trade_size": np.where(is_trade, size, 0),
        "size": size,
    })
    del bid, ask, mid, bid_sz, ask_sz, price, size, is_trade, trade_side, seconds, ts
    gc.collect()

    logger.info("  Grouping %d ticks into 1-sec bars...", len(df_reuse))

    agg = df_reuse.groupby("sec").agg(
        bid=("bid", "last"),
        ask=("ask", "last"),
        mid=("mid", "last"),
        bid_sz=("bid_sz", "last"),
        ask_sz=("ask_sz", "last"),
        trade_count=("is_trade", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        total_vol=("size", "sum"),
        last_trade_price=("trade_price", "last"),
        max_trade_size=("trade_size", "max"),
    )
    del df_reuse
    gc.collect()

    agg["spread_ticks"] = (agg["ask"] - agg["bid"]) / MES_TICK
    agg["log_return"] = np.log(agg["mid"] / agg["mid"].shift(1))

    # Forward-fill trade price for seconds without trades
    agg["last_trade_price"] = agg["last_trade_price"].ffill()

    return agg


def _compute_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute all 15 per-bar microstructure features. Returns [n_bars, 15].

    Computes the superset of features across all regimes. Each regime model
    selects its subset via REGIME_FEATURES indices.

    Features 0-4: common (all regimes)
    Features 5-8: mean-reverting / choppy (fade signals)
    Features 9-14: trending (momentum signals)
    """
    n = len(bars)
    n_feat = len(ALL_FEATURE_NAMES)
    features = np.zeros((n, n_feat), dtype=np.float32)

    bid = bars["bid"].values
    ask = bars["ask"].values
    mid = bars["mid"].values
    log_ret = bars["log_return"].values.astype(np.float32)
    trade_count = bars["trade_count"].values.astype(np.float32)
    buy_vol = bars["buy_vol"].values.astype(np.float32)
    sell_vol = bars["sell_vol"].values.astype(np.float32)
    total_vol = bars["total_vol"].values.astype(np.float32)
    last_trade_price = bars["last_trade_price"].values
    bid_sz = bars["bid_sz"].values.astype(np.float32)
    ask_sz = bars["ask_sz"].values.astype(np.float32)

    logger.info("  Computing %d features on %d 1-sec bars...", n_feat, n)

    # ── Common (0-4) ──
    # 0: Log return
    features[:, 0] = log_ret

    # 1: Spread in ticks
    features[:, 1] = bars["spread_ticks"].values

    # 2: Trade count
    features[:, 2] = trade_count

    # 3: Signed volume (buy - sell, raw)
    features[:, 3] = buy_vol - sell_vol

    # 4: Lee-Ready classification
    features[:, 4] = np.sign(last_trade_price - mid)

    # ── Mean-reverting / Choppy (5-8) ──
    # 5: Microprice displacement from mid (book lean)
    total_sz = bid_sz + ask_sz
    microprice = np.where(
        total_sz > 0,
        (bid * ask_sz + ask * bid_sz) / total_sz,
        mid,
    )
    features[:, 5] = (microprice - mid) / MES_TICK

    # 6: Trade imbalance — (buy - sell) / total
    features[:, 6] = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0.0)

    # 7: OFI — instantaneous bid/ask change
    bid_diff = np.diff(bid, prepend=bid[0])
    ask_diff = np.diff(ask, prepend=ask[0])
    ofi = bid_diff - ask_diff
    features[:, 7] = ofi

    # 8: Squared return (vol proxy)
    features[:, 8] = log_ret ** 2

    # ── Trending (9-14) ──
    # 9-10: Cumulative OFI over 5 and 15 bars
    ofi_cumsum = np.cumsum(ofi)
    features[:, 9] = ofi_cumsum - np.concatenate([[0] * 5, ofi_cumsum[:-5]])
    features[:, 10] = ofi_cumsum - np.concatenate([[0] * 15, ofi_cumsum[:-15]])

    # 11-12: Price velocity (cumulative return over 5 and 15 bars)
    ret_cumsum = np.cumsum(log_ret)
    features[:, 11] = ret_cumsum - np.concatenate([[0] * 5, ret_cumsum[:-5]])
    features[:, 12] = ret_cumsum - np.concatenate([[0] * 15, ret_cumsum[:-15]])

    # 13: Volume acceleration — current vs rolling mean(10)
    tc_cumsum = np.cumsum(trade_count)
    rolling_mean_10 = (tc_cumsum - np.concatenate([[0] * 10, tc_cumsum[:-10]])) / 10.0
    rolling_mean_10 = np.maximum(rolling_mean_10, 1.0)  # avoid div by zero
    features[:, 13] = trade_count / rolling_mean_10 - 1.0

    # 14: Book imbalance — bid_sz / (bid_sz + ask_sz)
    features[:, 14] = np.where(total_sz > 0, bid_sz / total_sz, 0.5)

    # Replace NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def _rs_for_chunk_size(windows: np.ndarray, cs: int) -> np.ndarray:
    """Vectorized R/S calculation for all windows at a given chunk size.

    Args:
        windows: [n_windows, window_size] array of returns
        cs: chunk size

    Returns:
        [n_windows] array of average R/S values (0 where invalid)
    """
    n_windows, window_size = windows.shape
    nc = window_size // cs

    # Reshape into chunks: [n_windows, nc, cs]
    trimmed = windows[:, :nc * cs].reshape(n_windows, nc, cs)

    # Chunk means and std: [n_windows, nc]
    chunk_means = trimmed.mean(axis=2)
    chunk_stds = trimmed.std(axis=2)

    # Deviations from chunk mean: [n_windows, nc, cs]
    dev = trimmed - chunk_means[:, :, np.newaxis]

    # Cumulative deviations: [n_windows, nc, cs]
    cumdev = np.cumsum(dev, axis=2)

    # R/S per chunk: (max - min) / std
    rs_range = cumdev.max(axis=2) - cumdev.min(axis=2)

    # Mask out chunks with near-zero std
    valid = chunk_stds > 1e-15
    rs_vals = np.where(valid, rs_range / np.where(valid, chunk_stds, 1.0), 0.0)

    # Average R/S across valid chunks per window
    valid_count = valid.sum(axis=1)
    rs_sum = rs_vals.sum(axis=1)
    avg_rs = np.where(valid_count > 0, rs_sum / valid_count, 0.0)

    return avg_rs


def _rolling_hurst(returns: np.ndarray, window: int = 200, step: int = 60) -> np.ndarray:
    """Rolling Hurst exponent via R/S method, computed every `step` bars and forward-filled."""
    n = len(returns)
    result = np.full(n, 0.5)

    # Collect all window positions
    positions = np.arange(window, n, step)
    if len(positions) == 0:
        return result

    # Extract all windows at once using stride tricks
    # Build index array: [n_positions, window]
    idx = positions[:, np.newaxis] - window + np.arange(window)[np.newaxis, :]
    windows = returns[idx]  # [n_positions, window]

    # Skip all-zero windows
    nonzero_mask = ~np.all(windows == 0, axis=1)

    # Precompute log(chunk_size) constants
    chunk_sizes = [cs for cs in [16, 32, 64, 128] if cs <= window]

    if len(chunk_sizes) < 2:
        return result

    # Compute avg R/S for each chunk size across all windows
    log_n_vals = np.log(np.array(chunk_sizes, dtype=np.float64))
    # [n_chunk_sizes, n_positions]
    log_rs_matrix = np.zeros((len(chunk_sizes), len(positions)))
    valid_matrix = np.zeros((len(chunk_sizes), len(positions)), dtype=bool)

    for ci, cs in enumerate(chunk_sizes):
        avg_rs = _rs_for_chunk_size(windows, cs)
        valid = (avg_rs > 0) & nonzero_mask
        log_rs_matrix[ci] = np.where(valid, np.log(np.where(valid, avg_rs, 1.0)), 0.0)
        valid_matrix[ci] = valid

    # Linear regression per window: H = slope of log(R/S) vs log(n)
    # Only use windows with >= 2 valid chunk sizes
    n_valid = valid_matrix.sum(axis=0)  # [n_positions]
    can_regress = n_valid >= 2

    for j in np.where(can_regress)[0]:
        mask = valid_matrix[:, j]
        x = log_n_vals[mask]
        y = log_rs_matrix[mask, j]
        nn = len(x)
        sx = x.sum()
        sy = y.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        denom = nn * sxx - sx * sx
        if abs(denom) > 1e-15:
            h = (nn * sxy - sx * sy) / denom
            h = max(0.0, min(1.0, h))
            pos = positions[j]
            fill_end = min(pos + step, n)
            result[pos:fill_end] = h

    return result


def _rolling_percentile(values: np.ndarray, window: int = 500) -> np.ndarray:
    """Rolling percentile rank using pandas rank."""
    s = pd.Series(values)
    roll_min = s.rolling(window, min_periods=1).min()
    roll_max = s.rolling(window, min_periods=1).max()
    denom = roll_max - roll_min
    result = np.where(denom > 0, (s - roll_min) / denom, 0.5)
    return np.nan_to_num(result, nan=0.0)


def _rolling_autocorr(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling lag-1 autocorrelation."""
    r = pd.Series(returns)
    r_lag = r.shift(1)
    # Use rolling correlation
    result = r.rolling(window, min_periods=3).corr(r_lag).fillna(0).values
    return result


def _z_score_normalize(features: np.ndarray, warmup: int = 100) -> np.ndarray:
    """Online z-score normalization using expanding window (no look-ahead)."""
    n, nf = features.shape
    result = np.zeros_like(features)

    for j in range(nf):
        col = features[:, j]
        cumsum = np.cumsum(col)
        cumsum2 = np.cumsum(col ** 2)
        counts = np.arange(1, n + 1, dtype=np.float64)

        mean = cumsum / counts
        var = cumsum2 / counts - mean ** 2
        var = np.maximum(var, 0)  # numerical safety
        std = np.sqrt(var)
        std[std < 1e-12] = 1.0

        result[:, j] = (col - mean) / std
        # Zero out warmup period
        result[:warmup, j] = 0.0

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _filter_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter 1-sec bars to RTH only (9:30-16:00 ET).

    Bars are indexed by unix timestamp (integer seconds). Converts to
    US/Eastern to determine RTH boundaries, then keeps only RTH bars.
    Also splits on session boundaries so overnight gaps don't bleed
    into feature windows.
    """
    # Convert unix seconds index to Eastern time
    utc_dt = pd.to_datetime(bars.index, unit="s", utc=True)
    et_dt = utc_dt.tz_convert("US/Eastern")

    # RTH: 9:30 - 16:00 ET
    time_of_day = et_dt.hour * 100 + et_dt.minute
    rth_mask = (time_of_day >= 930) & (time_of_day < 1600)

    bars_rth = bars.loc[rth_mask]
    logger.info("  RTH filter: %d -> %d bars (%.0f%% kept)",
                len(bars), len(bars_rth), 100 * len(bars_rth) / max(len(bars), 1))
    return bars_rth


def _resample_and_compute_chunked(
    l1_path: Path,
    output_dir: Path,
    output_name: str,
    rows_per_chunk: int = 5_000_000,
    rth_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream L1 parquet in chunks, resample to 1-sec bars, compute features.

    Processes ~5M ticks at a time (~200MB RAM) instead of 97M at once (~6GB).
    Writes intermediate 1-sec features and mid prices to mmap files.

    Args:
        rth_only: If True, filter to RTH hours (9:30-16:00 ET) before
                  computing features. Prevents overnight data from bleeding
                  into feature windows.

    Returns:
        norm_features: [n_bars, 9] float32 mmap
        mid: [n_bars] float64 mmap
    """
    needed_cols = ["timestamp", "bid_price", "ask_price", "bid_size", "ask_size", "price", "size"]
    pf = pq.ParquetFile(l1_path)
    total_rows = pf.metadata.num_rows
    logger.info("  Streaming %d ticks in chunks of %d", total_rows, rows_per_chunk)

    # First pass: resample each chunk to 1-sec bars, collect bar arrays
    all_bars_list: list[pd.DataFrame] = []
    rows_read = 0

    for batch in pf.iter_batches(batch_size=rows_per_chunk, columns=needed_cols):
        df_chunk = batch.to_pandas()
        rows_read += len(df_chunk)
        logger.info("  Resampling chunk: %d/%d ticks (%.0f%%)",
                     rows_read, total_rows, 100 * rows_read / total_rows)

        bars_chunk = _resample_to_1sec(df_chunk)
        del df_chunk
        gc.collect()

        all_bars_list.append(bars_chunk)

    # Merge 1-sec bars — much smaller than raw ticks (~300K bars vs 97M ticks)
    logger.info("  Merging %d bar chunks...", len(all_bars_list))
    bars = pd.concat(all_bars_list, axis=0)
    del all_bars_list
    gc.collect()

    # De-duplicate seconds that span chunk boundaries (keep last)
    bars = bars[~bars.index.duplicated(keep="last")]
    bars.sort_index(inplace=True)
    logger.info("  %d 1-sec bars after merge", len(bars))

    # Filter to RTH before computing features
    if rth_only:
        bars = _filter_rth(bars)

    # Detect session boundaries — gaps > 60s between consecutive bars
    # These mark overnight gaps where windows must not span across
    bar_seconds = bars.index.values
    gaps = np.diff(bar_seconds)
    session_breaks = np.where(gaps > 60)[0] + 1  # indices in bars where new sessions start
    logger.info("  %d session boundaries detected", len(session_breaks))

    # Compute features on the merged bars
    raw_features = _compute_features(bars)

    # Z-score normalize
    logger.info("  Normalizing features...")
    norm_features = _z_score_normalize(raw_features)
    del raw_features
    gc.collect()

    mid = bars["mid"].values.copy()
    del bars
    gc.collect()

    # Write to mmap so windowing step can read without holding in RAM
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bars = len(norm_features)

    feat_tmp_path = output_dir / f"{output_name}_bars_features.npy"
    mid_tmp_path = output_dir / f"{output_name}_bars_mid.npy"
    breaks_path = output_dir / f"{output_name}_session_breaks.npy"

    np.save(str(feat_tmp_path), norm_features)
    np.save(str(mid_tmp_path), mid)
    np.save(str(breaks_path), session_breaks)
    del norm_features, mid
    gc.collect()

    return (
        np.load(str(feat_tmp_path), mmap_mode="r"),
        np.load(str(mid_tmp_path), mmap_mode="r"),
        np.load(str(breaks_path)),
    )


def precompute_windows(
    l1_path: Path,
    window_size: int = 30,
    horizon_sec: int = 5,
    stride: int = 1,
    output_dir: Path | None = None,
    output_name: str = "windows",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process L1 parquet into labeled feature windows (vectorized).

    Streams L1 ticks in chunks (~5M at a time) to avoid loading the full
    dataset into RAM. Writes windows to mmap files on disk.

    Peak RAM usage: ~1-2GB instead of ~6GB for a full year of L1 data.

    Returns:
        features: [n_windows, window_size, 12] float32 (mmap if output_dir set)
        directions: [n_windows] float32 (0 or 1)
        magnitudes: [n_windows] float32 (ticks)
    """
    effective_output_dir = output_dir or Path("/tmp/l1_scalper_windows")

    # Step 1-3: Stream ticks, resample, compute features, normalize
    # This is the memory-critical part — chunked to ~1-2GB peak
    norm_features, mid, session_breaks = _resample_and_compute_chunked(
        l1_path, effective_output_dir, output_name,
    )

    # Step 4: Build sliding windows with labels (from mmap — cheap)
    logger.info("  Building windows (size=%d, stride=%d, horizon=%d)...",
                window_size, stride, horizon_sec)

    n_bars = len(norm_features)
    n_features = norm_features.shape[1]
    max_start = n_bars - window_size - horizon_sec

    if max_start <= 0:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    starts = np.arange(window_size, max_start, stride)

    # Remove windows that span session boundaries (overnight gaps)
    # A window starting at `s` uses bars [s - window_size, s + horizon_sec)
    # It's invalid if any session break falls within that range
    if len(session_breaks) > 0:
        valid = np.ones(len(starts), dtype=bool)
        for brk in session_breaks:
            # Window is invalid if break is within [start - window_size + 1, start + horizon_sec]
            invalid = (starts - window_size + 1 <= brk) & (brk <= starts + horizon_sec)
            valid &= ~invalid
        starts = starts[valid]
        logger.info("  %d windows after removing cross-session spans", len(starts))

    n_windows = len(starts)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        feat_path = output_dir / f"{output_name}_features.npy"
        # Create mmap file with correct shape
        feat_mmap = np.lib.format.open_memmap(
            str(feat_path), mode="w+",
            dtype=np.float32, shape=(n_windows, window_size, n_features),
        )
        # Write in chunks to limit RAM (~1000 windows at a time)
        chunk = 1000
        for i in range(0, n_windows, chunk):
            end = min(i + chunk, n_windows)
            batch_starts = starts[i:end]
            idx = batch_starts[:, np.newaxis] - window_size + np.arange(window_size)[np.newaxis, :]
            feat_mmap[i:end] = norm_features[idx]
        feat_mmap.flush()
        del feat_mmap
        gc.collect()

        # Labels are small — just compute in RAM and save
        mid_arr = np.array(mid)  # small read from mmap
        mid_now = mid_arr[starts - 1]
        mid_future = mid_arr[starts - 1 + horizon_sec]
        delta = mid_future - mid_now
        del mid_arr
        gc.collect()
        directions = np.where(delta > 0, 1.0, 0.0).astype(np.float32)
        magnitudes = (delta / MES_TICK).astype(np.float32)

        np.save(output_dir / f"{output_name}_directions.npy", directions)
        np.save(output_dir / f"{output_name}_magnitudes.npy", magnitudes)

        # Clean up intermediate bar-level files
        bars_feat_path = output_dir / f"{output_name}_bars_features.npy"
        bars_mid_path = output_dir / f"{output_name}_bars_mid.npy"
        breaks_path = output_dir / f"{output_name}_session_breaks.npy"
        del norm_features, mid
        gc.collect()
        bars_feat_path.unlink(missing_ok=True)
        bars_mid_path.unlink(missing_ok=True)
        breaks_path.unlink(missing_ok=True)

        logger.info("  %d windows written to mmap", n_windows)
        logger.info("  Direction balance: %.3f long / %.3f short",
                    directions.mean(), 1 - directions.mean())

        # Return mmap handles
        return (
            np.load(str(feat_path), mmap_mode="r"),
            np.load(str(output_dir / f"{output_name}_directions.npy"), mmap_mode="r"),
            np.load(str(output_dir / f"{output_name}_magnitudes.npy"), mmap_mode="r"),
        )

    # Fallback: in-memory (for small datasets / tests)
    window_idx = starts[:, np.newaxis] - window_size + np.arange(window_size)[np.newaxis, :]
    features_out = norm_features[window_idx]
    del norm_features
    gc.collect()

    mid_arr = np.array(mid)
    mid_now = mid_arr[starts - 1]
    mid_future = mid_arr[starts - 1 + horizon_sec]
    delta = mid_future - mid_now
    del mid_arr, mid
    gc.collect()
    directions = np.where(delta > 0, 1.0, 0.0).astype(np.float32)
    magnitudes = (delta / MES_TICK).astype(np.float32)

    logger.info("  %d labeled windows", n_windows)
    logger.info("  Direction balance: %.3f long / %.3f short",
                directions.mean(), 1 - directions.mean())

    return features_out, directions, magnitudes


def _bracket_exit_labels(
    mid: np.ndarray,
    starts: np.ndarray,
    tp_ticks: float = 9.0,
    sl_ticks: float = 9.0,
    max_hold_bars: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate bracket exit (TP/SL/timeout) for each entry window.

    For each entry at bar `starts[i]`, simulates a LONG trade:
    - TP hit if mid rises >= tp_ticks * MES_TICK from entry
    - SL hit if mid drops >= sl_ticks * MES_TICK from entry
    - Timeout after max_hold_bars (5 min at 1-sec bars)

    Returns:
        directions: [n_windows] float32 — 1.0 if trade was profitable, 0.0 otherwise
        magnitudes: [n_windows] float32 — realized P&L in ticks (positive or negative)

    Note: Labels assume LONG entry. The entry model predicts direction, so:
    - For windows where model predicts long: use labels as-is
    - For windows where model predicts short: labels would be flipped
    Since we train on direction prediction, we label based on whether the
    LONG side of the bracket would have been profitable. The model learns
    when long vs short is correct.
    """
    n_windows = len(starts)
    n_bars = len(mid)
    tp_dist = tp_ticks * MES_TICK
    sl_dist = sl_ticks * MES_TICK

    directions = np.zeros(n_windows, dtype=np.float32)
    magnitudes = np.zeros(n_windows, dtype=np.float32)

    # Vectorized: for each start, get the future mid prices
    # But max_hold_bars could be 300, and we have potentially millions of windows
    # Do it in batches to control memory

    batch_size = 10000
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        batch_starts = starts[batch_start:batch_end]
        b = len(batch_starts)

        # Entry prices
        entry_prices = mid[batch_starts - 1]

        # Future prices matrix: [batch, max_hold_bars]
        # Clip to available bars
        max_future = min(max_hold_bars, n_bars - batch_starts.max())
        if max_future <= 0:
            continue

        future_idx = batch_starts[:, np.newaxis] - 1 + np.arange(1, max_future + 1)[np.newaxis, :]
        # Clip indices to valid range
        future_idx = np.clip(future_idx, 0, n_bars - 1)
        future_prices = mid[future_idx]  # [batch, max_future]

        # Compute running P&L from entry
        pnl = future_prices - entry_prices[:, np.newaxis]  # [batch, max_future]

        # Find first TP hit (pnl >= tp_dist)
        tp_hit = pnl >= tp_dist
        tp_bar = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), max_future + 1)

        # Find first SL hit (pnl <= -sl_dist)
        sl_hit = pnl <= -sl_dist
        sl_bar = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), max_future + 1)

        # Determine outcome: whichever hits first
        tp_first = tp_bar < sl_bar
        sl_first = sl_bar < tp_bar
        timeout = ~tp_first & ~sl_first

        # Magnitudes
        batch_mag = np.zeros(b, dtype=np.float32)
        batch_mag[tp_first] = tp_ticks
        batch_mag[sl_first] = -sl_ticks
        # Timeout: use actual P&L at max_hold or end of data
        timeout_bar = np.minimum(max_future - 1, max_hold_bars - 1)
        timeout_idx = np.clip(timeout_bar, 0, max_future - 1)
        batch_mag[timeout] = (pnl[timeout, timeout_idx] / MES_TICK).astype(np.float32)

        # Direction: profitable = 1
        batch_dir = (batch_mag > 0).astype(np.float32)

        directions[batch_start:batch_end] = batch_dir
        magnitudes[batch_start:batch_end] = batch_mag

    return directions, magnitudes


def precompute_windows_bracket(
    l1_path: Path,
    window_size: int = 30,
    tp_ticks: float = 9.0,
    sl_ticks: float = 9.0,
    max_hold_bars: int = 300,
    stride: int = 1,
    output_dir: Path | None = None,
    output_name: str = "windows_bracket",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like precompute_windows but with bracket exit labels (TP/SL/timeout).

    Labels reflect realistic 9:9 bracket with 5-min timeout instead of
    simple forward-return at a fixed horizon.

    Returns:
        features: [n_windows, window_size, n_features] float32
        directions: [n_windows] float32 — 1.0 if bracket trade profitable
        magnitudes: [n_windows] float32 — realized P&L in ticks
    """
    effective_output_dir = output_dir or Path("/tmp/l1_scalper_windows")

    # Reuse the chunked resample + feature pipeline
    norm_features, mid, session_breaks = _resample_and_compute_chunked(
        l1_path, effective_output_dir, output_name,
    )

    # Need enough future bars for bracket simulation
    min_future = max_hold_bars
    logger.info("  Building bracket windows (size=%d, stride=%d, TP=%g, SL=%g, maxhold=%d)...",
                window_size, stride, tp_ticks, sl_ticks, max_hold_bars)

    n_bars = len(norm_features)
    n_features = norm_features.shape[1]
    max_start = n_bars - window_size - min_future

    if max_start <= 0:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    starts = np.arange(window_size, max_start, stride)

    # Remove cross-session windows
    if len(session_breaks) > 0:
        valid = np.ones(len(starts), dtype=bool)
        for brk in session_breaks:
            # Window + bracket hold must not span a session break
            invalid = (starts - window_size + 1 <= brk) & (brk <= starts + min_future)
            valid &= ~invalid
        starts = starts[valid]
        logger.info("  %d windows after removing cross-session spans", len(starts))

    n_windows = len(starts)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        feat_path = output_dir / f"{output_name}_features.npy"
        feat_mmap = np.lib.format.open_memmap(
            str(feat_path), mode="w+",
            dtype=np.float32, shape=(n_windows, window_size, n_features),
        )
        chunk = 1000
        for i in range(0, n_windows, chunk):
            end = min(i + chunk, n_windows)
            batch_starts = starts[i:end]
            idx = batch_starts[:, np.newaxis] - window_size + np.arange(window_size)[np.newaxis, :]
            feat_mmap[i:end] = norm_features[idx]
        feat_mmap.flush()
        del feat_mmap
        gc.collect()

        # Bracket exit labels
        mid_arr = np.array(mid)
        directions, magnitudes = _bracket_exit_labels(
            mid_arr, starts, tp_ticks=tp_ticks, sl_ticks=sl_ticks, max_hold_bars=max_hold_bars,
        )
        del mid_arr
        gc.collect()

        np.save(output_dir / f"{output_name}_directions.npy", directions)
        np.save(output_dir / f"{output_name}_magnitudes.npy", magnitudes)

        # Cleanup intermediate files
        for suffix in ["_bars_features.npy", "_bars_mid.npy", "_session_breaks.npy"]:
            (output_dir / f"{output_name}{suffix}").unlink(missing_ok=True)
        del norm_features, mid
        gc.collect()

        logger.info("  %d bracket windows written", n_windows)
        logger.info("  Win rate: %.3f, avg magnitude: %.2f ticks",
                    directions.mean(), magnitudes.mean())

        return (
            np.load(str(feat_path), mmap_mode="r"),
            np.load(str(output_dir / f"{output_name}_directions.npy"), mmap_mode="r"),
            np.load(str(output_dir / f"{output_name}_magnitudes.npy"), mmap_mode="r"),
        )

    # In-memory fallback
    window_idx = starts[:, np.newaxis] - window_size + np.arange(window_size)[np.newaxis, :]
    features_out = norm_features[window_idx]
    del norm_features
    gc.collect()

    mid_arr = np.array(mid)
    directions, magnitudes = _bracket_exit_labels(
        mid_arr, starts, tp_ticks=tp_ticks, sl_ticks=sl_ticks, max_hold_bars=max_hold_bars,
    )
    del mid_arr, mid
    gc.collect()

    logger.info("  %d bracket windows", n_windows)
    logger.info("  Win rate: %.3f, avg magnitude: %.2f ticks",
                directions.mean(), magnitudes.mean())

    return features_out, directions, magnitudes


class EntryDataset(Dataset):
    """PyTorch dataset for CNN-LSTM entry model.

    Supports both in-memory numpy arrays and memory-mapped .npy files.
    Optional feature_indices to select a subset of feature columns
    (for regime-specific models).
    """

    def __init__(
        self,
        features: np.ndarray,
        directions: np.ndarray,
        magnitudes: np.ndarray,
        feature_indices: list[int] | None = None,
    ) -> None:
        # Keep as numpy — convert per-batch in __getitem__ to avoid doubling RAM
        self.features = features
        self.directions = directions
        self.magnitudes = magnitudes
        self.feature_indices = feature_indices

    @classmethod
    def from_mmap(
        cls,
        features_path: Path,
        directions_path: Path,
        magnitudes_path: Path,
        feature_indices: list[int] | None = None,
    ) -> "EntryDataset":
        """Load dataset from memory-mapped .npy files (near-zero RAM)."""
        features = np.load(features_path, mmap_mode="r")
        directions = np.load(directions_path, mmap_mode="r")
        magnitudes = np.load(magnitudes_path, mmap_mode="r")
        return cls(features, directions, magnitudes, feature_indices)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = np.array(self.features[idx])
        if self.feature_indices is not None:
            feat = feat[:, self.feature_indices]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(np.array(self.directions[idx])),
            torch.from_numpy(np.array(self.magnitudes[idx])),
        )


class IndexedEntryDataset(Dataset):
    """PyTorch dataset that reads from mmap by index — zero copy.

    Instead of copying filtered arrays into RAM, stores integer indices
    into the original mmap arrays. Each __getitem__ reads one window
    from disk via mmap page cache. Much more memory efficient for large
    filtered subsets.
    """

    def __init__(
        self,
        features: np.ndarray,
        directions: np.ndarray,
        magnitudes: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.directions = directions
        self.magnitudes = magnitudes
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        feat = np.array(self.features[real_idx])
        return (
            torch.from_numpy(feat),
            torch.tensor(self.directions[real_idx], dtype=torch.float32),
            torch.tensor(self.magnitudes[real_idx], dtype=torch.float32),
        )


def save_windows_mmap(
    cache_dir: Path,
    name: str,
    features: np.ndarray,
    directions: np.ndarray,
    magnitudes: np.ndarray,
) -> None:
    """Save precomputed windows as individual .npy files for mmap loading."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{name}_features.npy", features)
    np.save(cache_dir / f"{name}_directions.npy", directions)
    np.save(cache_dir / f"{name}_magnitudes.npy", magnitudes)


def load_windows_mmap(
    cache_dir: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed windows as memory-mapped arrays."""
    return (
        np.load(cache_dir / f"{name}_features.npy", mmap_mode="r"),
        np.load(cache_dir / f"{name}_directions.npy", mmap_mode="r"),
        np.load(cache_dir / f"{name}_magnitudes.npy", mmap_mode="r"),
    )
