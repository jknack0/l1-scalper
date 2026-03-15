"""Cumulative Volume Delta with Lee-Ready classification."""

from __future__ import annotations

from collections import deque
from math import isnan

from src.data.schemas import L1Record
from src.features.base import Feature
from src.features.lee_ready import classify_trade

# 300 seconds in nanoseconds
DEFAULT_WINDOW_NS = 300_000_000_000


class CumulativeVolumeDelta(Feature):
    """Rolling CVD: sum(buy_volume) - sum(sell_volume) over a time window."""

    name = "cvd"

    def __init__(self, window_ns: int = DEFAULT_WINDOW_NS) -> None:
        self._window_ns = window_ns
        self._events: deque[tuple[int, int]] = deque()  # (timestamp, signed_volume)
        self._cvd: int = 0
        self._prev_trade_price: float | None = None

    def update(self, record: L1Record) -> float | None:
        ts = record.timestamp

        # Expire old events
        cutoff = ts - self._window_ns
        while self._events and self._events[0][0] < cutoff:
            _, old_vol = self._events.popleft()
            self._cvd -= old_vol

        if record.event_type != "trade":
            return float(self._cvd) if self._events else None

        if isnan(record.trade_price) or record.trade_size == 0:
            return float(self._cvd) if self._events else None

        side = classify_trade(
            record.trade_price,
            record.bid_price,
            record.ask_price,
            self._prev_trade_price,
        )
        self._prev_trade_price = record.trade_price

        signed_vol = side * record.trade_size
        self._events.append((ts, signed_vol))
        self._cvd += signed_vol

        return float(self._cvd)

    def reset(self) -> None:
        self._events.clear()
        self._cvd = 0
        self._prev_trade_price = None
