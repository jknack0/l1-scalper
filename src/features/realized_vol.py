"""Rolling realized volatility from 1-second log returns."""

from __future__ import annotations

from collections import deque
from math import log, sqrt

from src.data.schemas import L1Record
from src.features.base import Feature

# 60 seconds in nanoseconds
DEFAULT_WINDOW_NS = 60_000_000_000
# 1 second in nanoseconds
ONE_SEC_NS = 1_000_000_000


class RealizedVolatility(Feature):
    """Standard deviation of 1-second mid-price log returns over rolling 60s window."""

    name = "realized_vol"

    def __init__(self, window_ns: int = DEFAULT_WINDOW_NS) -> None:
        self._window_ns = window_ns
        self._returns: deque[tuple[int, float]] = deque()  # (second_ts, log_return)
        self._last_mid: float = 0.0
        self._last_second: int = 0
        self._current_second_mid: float = 0.0

    def update(self, record: L1Record) -> float | None:
        mid = (record.bid_price + record.ask_price) / 2.0
        ts = record.timestamp
        second = ts // ONE_SEC_NS

        # Track the latest mid per second
        self._current_second_mid = mid

        # New second boundary — emit a return
        if second != self._last_second and self._last_second > 0 and self._last_mid > 0:
            if mid > 0 and self._last_mid > 0:
                log_ret = log(mid / self._last_mid)
                self._returns.append((ts, log_ret))

            self._last_mid = mid
            self._last_second = second
        elif self._last_second == 0:
            self._last_mid = mid
            self._last_second = second

        # Expire old returns
        cutoff = ts - self._window_ns
        while self._returns and self._returns[0][0] < cutoff:
            self._returns.popleft()

        if len(self._returns) < 2:
            return None

        # Compute std of returns
        rets = [r for _, r in self._returns]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        return sqrt(var)

    def reset(self) -> None:
        self._returns.clear()
        self._last_mid = 0.0
        self._last_second = 0
        self._current_second_mid = 0.0
