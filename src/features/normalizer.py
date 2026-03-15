"""Welford's online z-score normalizer and staleness detection."""

from __future__ import annotations

import math
from collections import deque


class WelfordNormalizer:
    """Online z-score normalizer using Welford's algorithm over a rolling window.

    Uses a circular buffer of `window` values. Returns z-score = (x - mean) / std.
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._buffer: deque[float] = deque(maxlen=window)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared differences from mean

    def update(self, value: float) -> float:
        """Add a value and return its z-score."""
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # If buffer is full, remove the oldest value
        if len(self._buffer) == self._window:
            old = self._buffer[0]
            self._remove(old)

        self._add(value)
        self._buffer.append(value)

        if self._count < 2:
            return 0.0

        std = math.sqrt(self._m2 / self._count)
        if std < 1e-12:
            return 0.0

        return (value - self._mean) / std

    def is_valid(self) -> bool:
        """True if we have enough samples for meaningful normalization."""
        return self._count >= 20

    def _add(self, value: float) -> None:
        """Welford online add."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _remove(self, value: float) -> None:
        """Reverse Welford step to remove the oldest value."""
        if self._count <= 1:
            self._count = 0
            self._mean = 0.0
            self._m2 = 0.0
            return
        old_mean = self._mean
        self._count -= 1
        self._mean = (old_mean * (self._count + 1) - value) / self._count
        # Reverse M2 update
        delta = value - self._mean
        delta2 = value - old_mean
        self._m2 -= delta * delta2
        # Guard against floating-point drift
        if self._m2 < 0:
            self._m2 = 0.0

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        return math.sqrt(self._m2 / self._count)


def staleness_gate(value: float, current_ts: int, last_ts: int, max_age_ns: int = 2_000_000_000) -> float:
    """Return 0.0 if last update was > max_age_ns ago, else return value."""
    if last_ts == 0:
        return 0.0
    if (current_ts - last_ts) > max_age_ns:
        return 0.0
    return value
