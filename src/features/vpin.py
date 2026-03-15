"""Volume-Synchronized Probability of Informed Trading (VPIN).

Buckets trades into fixed-volume bins, classifies via Lee-Ready,
then computes rolling |buy_vol - sell_vol| / bucket_size.
"""

from __future__ import annotations

from collections import deque
from math import isnan

from src.data.schemas import L1Record
from src.features.base import Feature
from src.features.lee_ready import classify_trade


class VPIN(Feature):
    """VPIN computed over rolling volume buckets.

    bucket_size: volume per bucket (default 500, tune to ~median 1-min volume)
    num_buckets: number of completed buckets for rolling average (default 50)
    """

    name = "vpin"

    def __init__(self, bucket_size: int = 500, num_buckets: int = 50) -> None:
        self._bucket_size = bucket_size
        self._num_buckets = num_buckets
        self._buy_vol = 0
        self._sell_vol = 0
        self._bucket_vol = 0
        self._completed: deque[float] = deque(maxlen=num_buckets)
        self._prev_trade_price: float | None = None

    def update(self, record: L1Record) -> float | None:
        if record.event_type != "trade":
            return self._current_vpin()

        if isnan(record.trade_price) or record.trade_size == 0:
            return self._current_vpin()

        side = classify_trade(
            record.trade_price,
            record.bid_price,
            record.ask_price,
            self._prev_trade_price,
        )
        self._prev_trade_price = record.trade_price

        size = record.trade_size
        remaining = size

        while remaining > 0:
            space = self._bucket_size - self._bucket_vol
            fill = min(remaining, space)

            if side >= 0:
                self._buy_vol += fill
            else:
                self._sell_vol += fill
            self._bucket_vol += fill
            remaining -= fill

            if self._bucket_vol >= self._bucket_size:
                imbalance = abs(self._buy_vol - self._sell_vol) / self._bucket_size
                self._completed.append(imbalance)
                self._buy_vol = 0
                self._sell_vol = 0
                self._bucket_vol = 0

        return self._current_vpin()

    def _current_vpin(self) -> float | None:
        if len(self._completed) == 0:
            return None
        return sum(self._completed) / len(self._completed)

    def reset(self) -> None:
        self._buy_vol = 0
        self._sell_vol = 0
        self._bucket_vol = 0
        self._completed.clear()
        self._prev_trade_price = None
