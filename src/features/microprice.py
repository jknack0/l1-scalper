"""Stoikov micro-price from bid/ask imbalance."""

from __future__ import annotations

from math import isnan, log

from src.data.schemas import L1Record
from src.features.base import Feature


class MicroPrice(Feature):
    """Micro-price log-return feature.

    micro = mid + spread * (bid_size - ask_size) / (2 * (bid_size + ask_size))
    Feature = log(micro_now / micro_prev)
    """

    name = "microprice"

    def __init__(self) -> None:
        self._prev_micro: float = 0.0
        self._has_prev: bool = False

    def update(self, record: L1Record) -> float | None:
        bid_sz = record.bid_size
        ask_sz = record.ask_size
        total_sz = bid_sz + ask_sz

        if total_sz == 0:
            return None

        mid = (record.bid_price + record.ask_price) / 2.0
        spread = record.ask_price - record.bid_price
        micro = mid + spread * (bid_sz - ask_sz) / (2.0 * total_sz)

        if micro <= 0:
            return None

        if not self._has_prev:
            self._prev_micro = micro
            self._has_prev = True
            return None

        log_ret = log(micro / self._prev_micro) if self._prev_micro > 0 else 0.0
        self._prev_micro = micro
        return log_ret

    def reset(self) -> None:
        self._prev_micro = 0.0
        self._has_prev = False
