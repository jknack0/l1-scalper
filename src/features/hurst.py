"""Rolling Hurst exponent via rescaled range (R/S) method."""

from __future__ import annotations

from collections import deque
from math import isnan

import numpy as np

from src.data.schemas import L1Record
from src.features.base import Feature


class HurstExponent(Feature):
    """Rolling Hurst exponent over last N trade prices.

    H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
    Uses the rescaled range (R/S) method.
    """

    name = "hurst"

    def __init__(self, window: int = 200) -> None:
        self._window = window
        self._prices: deque[float] = deque(maxlen=window)

    def update(self, record: L1Record) -> float | None:
        if record.event_type != "trade":
            return self._compute() if len(self._prices) >= self._window else None

        if isnan(record.trade_price):
            return self._compute() if len(self._prices) >= self._window else None

        self._prices.append(record.trade_price)

        if len(self._prices) < self._window:
            return None

        return self._compute()

    def _compute(self) -> float | None:
        if len(self._prices) < self._window:
            return None

        prices = np.array(self._prices)

        # Vectorized log returns (filter positive prices)
        valid = (prices[:-1] > 0) & (prices[1:] > 0)
        if valid.sum() < 10:
            return None
        returns = np.log(prices[1:][valid] / prices[:-1][valid])

        # R/S calculation over multiple sub-series lengths
        log_rs = []
        log_n = []
        n_ret = len(returns)

        for chunk_size in [16, 32, 64, 128]:
            if chunk_size > n_ret:
                break

            num_chunks = n_ret // chunk_size
            # Reshape into [num_chunks, chunk_size]
            trimmed = returns[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)

            # Vectorized across all chunks
            stds = trimmed.std(axis=1)
            devs = trimmed - trimmed.mean(axis=1, keepdims=True)
            cumdevs = np.cumsum(devs, axis=1)
            rs_range = cumdevs.max(axis=1) - cumdevs.min(axis=1)

            valid_mask = stds > 1e-15
            if valid_mask.any():
                avg_rs = float((rs_range[valid_mask] / stds[valid_mask]).mean())
                if avg_rs > 0:
                    log_rs.append(np.log(avg_rs))
                    log_n.append(np.log(chunk_size))

        if len(log_rs) < 2:
            return 0.5

        # Linear regression: H = slope of log(R/S) vs log(n)
        x = np.array(log_n)
        y = np.array(log_rs)
        n_pts = len(x)
        denom = n_pts * (x * x).sum() - x.sum() ** 2
        if abs(denom) < 1e-15:
            return 0.5

        hurst = float((n_pts * (x * y).sum() - x.sum() * y.sum()) / denom)
        return max(0.0, min(1.0, hurst))

    def reset(self) -> None:
        self._prices.clear()
