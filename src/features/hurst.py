"""Rolling Hurst exponent via rescaled range (R/S) method."""

from __future__ import annotations

from collections import deque
from math import isnan, log, sqrt

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

        prices = list(self._prices)
        n = len(prices)

        # Log returns
        returns = []
        for i in range(1, n):
            if prices[i - 1] > 0 and prices[i] > 0:
                returns.append(log(prices[i] / prices[i - 1]))

        if len(returns) < 10:
            return None

        # R/S calculation over multiple sub-series lengths
        log_rs = []
        log_n = []

        for chunk_size in [16, 32, 64, 128]:
            if chunk_size > len(returns):
                break

            rs_values = []
            num_chunks = len(returns) // chunk_size

            for i in range(num_chunks):
                chunk = returns[i * chunk_size : (i + 1) * chunk_size]
                mean_r = sum(chunk) / len(chunk)
                deviations = [r - mean_r for r in chunk]

                # Cumulative deviations
                cumsum = []
                s = 0.0
                for d in deviations:
                    s += d
                    cumsum.append(s)

                r_range = max(cumsum) - min(cumsum)
                std = sqrt(sum(d * d for d in deviations) / len(deviations))

                if std > 1e-15:
                    rs_values.append(r_range / std)

            if rs_values:
                avg_rs = sum(rs_values) / len(rs_values)
                if avg_rs > 0:
                    log_rs.append(log(avg_rs))
                    log_n.append(log(chunk_size))

        if len(log_rs) < 2:
            return 0.5  # default to random walk

        # Linear regression: H = slope of log(R/S) vs log(n)
        n_pts = len(log_rs)
        sum_x = sum(log_n)
        sum_y = sum(log_rs)
        sum_xy = sum(x * y for x, y in zip(log_n, log_rs))
        sum_xx = sum(x * x for x in log_n)

        denom = n_pts * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-15:
            return 0.5

        hurst = (n_pts * sum_xy - sum_x * sum_y) / denom
        return max(0.0, min(1.0, hurst))  # clamp to [0, 1]

    def reset(self) -> None:
        self._prices.clear()
