"""Streaming feature pipeline for live trading.

Converts raw L1 ticks into 1-second bars with all 15 features,
normalized using Welford's online z-score. Maintains a rolling
window buffer for model inference.

Designed to be called tick-by-tick from the live bot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MES_TICK = 0.25


@dataclass
class BarAccumulator:
    """Accumulates ticks within a 1-second bar."""
    second: int = 0  # unix second
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    bid_sz: float = 0.0
    ask_sz: float = 0.0
    trade_count: int = 0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    total_vol: float = 0.0
    last_trade_price: float = 0.0
    n_ticks: int = 0

    def update(self, tick: dict) -> None:
        self.bid = tick["bid_price"]
        self.ask = tick["ask_price"]
        self.mid = (self.bid + self.ask) / 2.0
        self.bid_sz = tick.get("bid_size", 1)
        self.ask_sz = tick.get("ask_size", 1)

        trade_size = tick.get("trade_size", 0) or 0
        if trade_size > 0:
            self.trade_count += 1
            self.total_vol += trade_size
            self.last_trade_price = tick.get("trade_price", self.mid)
            side = tick.get("trade_side", 0)
            if side > 0:
                self.buy_vol += trade_size
            elif side < 0:
                self.sell_vol += trade_size

        self.n_ticks += 1


class WelfordNormalizer:
    """Online z-score normalization using Welford's algorithm.

    Maintains per-feature running mean and variance. Clips to ±5σ.
    """

    def __init__(self, n_features: int, warmup: int = 100) -> None:
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
        self.warmup = warmup

    def update_and_normalize(self, features: np.ndarray) -> np.ndarray:
        """Update stats with new observation and return z-scored version."""
        self.n += 1
        delta = features - self.mean
        self.mean += delta / self.n
        delta2 = features - self.mean
        self.m2 += delta * delta2

        if self.n < self.warmup:
            return np.zeros_like(features, dtype=np.float32)

        var = self.m2 / self.n
        std = np.sqrt(np.maximum(var, 1e-10))
        z = (features - self.mean) / std
        return np.clip(z, -5.0, 5.0).astype(np.float32)


class StreamingFeatures:
    """Streaming feature pipeline: ticks → 1-sec bars → 15 features → z-score → rolling window.

    Usage:
        sf = StreamingFeatures(window_size=30)
        for tick in live_feed:
            bar = sf.on_tick(tick)
            if bar is not None:
                window = sf.get_window()  # [window_size, 15] or None if not enough bars
    """

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._normalizer = WelfordNormalizer(n_features=15)

        # Rolling window of normalized features
        self._buffer = np.zeros((window_size * 2, 15), dtype=np.float32)  # 2x for efficiency
        self._buffer_len = 0

        # Current bar accumulator
        self._current_bar = BarAccumulator()
        self._current_second = 0

        # Previous bar values for feature computation
        self._prev_mid = 0.0
        self._prev_bid = 0.0
        self._prev_ask = 0.0
        self._prev_bid_sz = 0.0
        self._prev_ask_sz = 0.0

        # Rolling accumulators for cumulative features
        self._ofi_history: list[float] = []
        self._ret_history: list[float] = []
        self._tc_history: list[float] = []

        self._bars_processed = 0

        # Latest bar data for position manager
        self.latest_mid: float = 0.0
        self.latest_bid: float = 0.0
        self.latest_ask: float = 0.0

    def on_tick(self, tick: dict) -> bool:
        """Process one tick. Returns True if a new bar was completed.

        tick should have: timestamp (ns), bid_price, ask_price, bid_size, ask_size,
        trade_price, trade_size, trade_side
        """
        ts_ns = tick["timestamp"]
        second = int(ts_ns // 1_000_000_000)

        if self._current_second == 0:
            self._current_second = second
            self._current_bar = BarAccumulator(second=second)

        if second != self._current_second:
            # New second — finalize current bar
            self._finalize_bar()
            self._current_second = second
            self._current_bar = BarAccumulator(second=second)

        self._current_bar.update(tick)
        return False

    def _finalize_bar(self) -> None:
        """Convert accumulated ticks into a feature vector and add to buffer."""
        bar = self._current_bar
        if bar.n_ticks == 0:
            return

        mid = bar.mid
        self.latest_mid = mid
        self.latest_bid = bar.bid
        self.latest_ask = bar.ask

        # Compute log return
        if self._prev_mid > 0:
            log_ret = np.log(mid / self._prev_mid) if self._prev_mid > 0 else 0.0
        else:
            log_ret = 0.0

        spread_ticks = (bar.ask - bar.bid) / MES_TICK

        # OFI
        bid_diff = bar.bid_sz - self._prev_bid_sz if self._prev_bid_sz > 0 else 0.0
        ask_diff = bar.ask_sz - self._prev_ask_sz if self._prev_ask_sz > 0 else 0.0
        ofi = bid_diff - ask_diff

        # Microprice
        total_sz = bar.bid_sz + bar.ask_sz
        if total_sz > 0:
            microprice = (bar.bid * bar.ask_sz + bar.ask * bar.bid_sz) / total_sz
        else:
            microprice = mid
        microprice_disp = (microprice - mid) / MES_TICK

        # Trade imbalance
        if bar.total_vol > 0:
            trade_imbalance = (bar.buy_vol - bar.sell_vol) / bar.total_vol
        else:
            trade_imbalance = 0.0

        # Lee-Ready
        lee_ready = np.sign(bar.last_trade_price - mid) if bar.last_trade_price > 0 else 0.0

        # Book imbalance
        if total_sz > 0:
            book_imbalance = bar.bid_sz / total_sz
        else:
            book_imbalance = 0.5

        # Update rolling histories
        self._ofi_history.append(ofi)
        self._ret_history.append(log_ret)
        self._tc_history.append(bar.trade_count)

        # Keep last 30 bars for rolling features
        max_hist = 30
        if len(self._ofi_history) > max_hist:
            self._ofi_history = self._ofi_history[-max_hist:]
            self._ret_history = self._ret_history[-max_hist:]
            self._tc_history = self._tc_history[-max_hist:]

        # Cumulative OFI (5, 15 bars)
        ofi_arr = self._ofi_history
        cum_ofi_5 = sum(ofi_arr[-5:]) if len(ofi_arr) >= 5 else sum(ofi_arr)
        cum_ofi_15 = sum(ofi_arr[-15:]) if len(ofi_arr) >= 15 else sum(ofi_arr)

        # Price velocity (5, 15 bars)
        ret_arr = self._ret_history
        vel_5 = sum(ret_arr[-5:]) if len(ret_arr) >= 5 else sum(ret_arr)
        vel_15 = sum(ret_arr[-15:]) if len(ret_arr) >= 15 else sum(ret_arr)

        # Volume acceleration
        tc_arr = self._tc_history
        if len(tc_arr) >= 10:
            rolling_mean_10 = sum(tc_arr[-10:]) / 10.0
            vol_accel = bar.trade_count / max(rolling_mean_10, 1.0) - 1.0
        else:
            vol_accel = 0.0

        # Build feature vector (same order as dataset.py ALL_FEATURE_NAMES)
        features = np.array([
            log_ret,                    # 0: log_return
            spread_ticks,               # 1: spread
            bar.trade_count,            # 2: trade_count
            bar.buy_vol - bar.sell_vol, # 3: signed_volume
            lee_ready,                  # 4: lee_ready
            microprice_disp,            # 5: microprice_displace
            trade_imbalance,            # 6: trade_imbalance
            ofi,                        # 7: ofi
            log_ret ** 2,               # 8: log_return_sq
            cum_ofi_5,                  # 9: cum_ofi_5
            cum_ofi_15,                 # 10: cum_ofi_15
            vel_5,                      # 11: price_velocity_5
            vel_15,                     # 12: price_velocity_15
            vol_accel,                  # 13: volume_accel
            book_imbalance,             # 14: book_imbalance
        ], dtype=np.float64)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score normalize
        normalized = self._normalizer.update_and_normalize(features)

        # Add to rolling buffer
        if self._buffer_len >= self.window_size * 2:
            # Shift buffer down
            self._buffer[:self.window_size] = self._buffer[self.window_size:]
            self._buffer_len = self.window_size

        self._buffer[self._buffer_len] = normalized
        self._buffer_len += 1

        # Save previous values
        self._prev_mid = mid
        self._prev_bid = bar.bid
        self._prev_ask = bar.ask
        self._prev_bid_sz = bar.bid_sz
        self._prev_ask_sz = bar.ask_sz

        self._bars_processed += 1

    def get_window(self) -> np.ndarray | None:
        """Get the current [window_size, 15] feature window, or None if not enough bars."""
        if self._buffer_len < self.window_size:
            return None
        start = self._buffer_len - self.window_size
        return self._buffer[start:self._buffer_len].copy()

    @property
    def bars_ready(self) -> bool:
        return self._buffer_len >= self.window_size

    @property
    def n_bars(self) -> int:
        return self._bars_processed
