"""Rolling percentile rank of trade size."""

from __future__ import annotations

import bisect
from collections import deque
from math import isnan

from src.data.schemas import L1Record
from src.features.base import Feature


class TradeSizeDistribution(Feature):
    """Rolling percentile rank of current trade size vs last N trades.

    Output: 0.0 to 1.0 (0.95 = 95th percentile of recent sizes).
    """

    name = "trade_size_dist"

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._sizes: deque[int] = deque(maxlen=window)
        self._sorted: list[int] = []

    def update(self, record: L1Record) -> float | None:
        if record.event_type != "trade":
            return None

        if record.trade_size == 0:
            return None

        size = record.trade_size

        # Remove oldest if at capacity
        if len(self._sizes) == self._window:
            old = self._sizes[0]
            idx = bisect.bisect_left(self._sorted, old)
            if idx < len(self._sorted) and self._sorted[idx] == old:
                self._sorted.pop(idx)

        self._sizes.append(size)
        bisect.insort(self._sorted, size)

        if len(self._sorted) < 2:
            return None

        # Percentile rank
        rank = bisect.bisect_right(self._sorted, size)
        return rank / len(self._sorted)

    def reset(self) -> None:
        self._sizes.clear()
        self._sorted.clear()
