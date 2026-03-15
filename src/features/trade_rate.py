"""Trade arrival rate — trades per second over a rolling window."""

from __future__ import annotations

from collections import deque

from src.data.schemas import L1Record
from src.features.base import Feature

# 30 seconds in nanoseconds
DEFAULT_WINDOW_NS = 30_000_000_000


class TradeRate(Feature):
    """Rolling trade count / window_seconds."""

    name = "trade_rate"

    def __init__(self, window_ns: int = DEFAULT_WINDOW_NS) -> None:
        self._window_ns = window_ns
        self._timestamps: deque[int] = deque()

    def update(self, record: L1Record) -> float | None:
        ts = record.timestamp

        # Expire old trades
        cutoff = ts - self._window_ns
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if record.event_type == "trade":
            self._timestamps.append(ts)

        window_sec = self._window_ns / 1e9
        return len(self._timestamps) / window_sec

    def reset(self) -> None:
        self._timestamps.clear()
