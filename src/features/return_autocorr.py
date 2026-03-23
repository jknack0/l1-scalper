"""Rolling lag-1 autocorrelation of 1-second mid-price returns."""

from __future__ import annotations

from collections import deque
from math import log

import numpy as np

from src.data.schemas import L1Record
from src.features.base import Feature

ONE_SEC_NS = 1_000_000_000
DEFAULT_WINDOW = 60  # 60 one-second returns


class ReturnAutocorrelation(Feature):
    """Lag-1 autocorrelation of 1-second mid-price returns over rolling 60s.

    Range: -1 to 1. Positive = momentum, negative = mean-reversion.
    """

    name = "return_autocorr"

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self._window = window
        self._returns: deque[float] = deque(maxlen=window)
        self._last_mid: float = 0.0
        self._last_second: int = 0

    def update(self, record: L1Record) -> float | None:
        mid = (record.bid_price + record.ask_price) / 2.0
        ts = record.timestamp
        second = ts // ONE_SEC_NS

        if second != self._last_second and self._last_second > 0:
            if mid > 0 and self._last_mid > 0:
                log_ret = log(mid / self._last_mid)
                self._returns.append(log_ret)
            self._last_mid = mid
            self._last_second = second
        elif self._last_second == 0:
            self._last_mid = mid
            self._last_second = second

        if len(self._returns) < 3:
            return None

        return self._compute()

    def _compute(self) -> float:
        """Lag-1 autocorrelation via NumPy vectorized Pearson formula."""
        rets = np.array(self._returns)

        x = rets[:-1]
        y = rets[1:]

        dx = x - x.mean()
        dy = y - y.mean()

        denom = np.sqrt((dx * dx).sum() * (dy * dy).sum())
        if denom < 1e-15:
            return 0.0

        corr = float((dx * dy).sum() / denom)
        return max(-1.0, min(1.0, corr))

    def reset(self) -> None:
        self._returns.clear()
        self._last_mid = 0.0
        self._last_second = 0
