"""Lee-Ready trade classification: buyer vs seller initiated."""

from __future__ import annotations

from math import isnan

from src.data.schemas import L1Record
from src.features.base import Feature


class LeeReadyClassifier(Feature):
    """Classify trades as buyer-initiated (+1) or seller-initiated (-1).

    - trade_price > mid -> buyer (+1)
    - trade_price < mid -> seller (-1)
    - trade_price == mid -> tick rule (compare to previous trade price)
    """

    name = "lee_ready"

    def __init__(self) -> None:
        self._prev_trade_price: float = 0.0
        self._has_prev: bool = False

    def update(self, record: L1Record) -> float | None:
        if record.event_type != "trade":
            return None

        if isnan(record.trade_price):
            return None

        mid = (record.bid_price + record.ask_price) / 2.0
        trade_price = record.trade_price

        if trade_price > mid:
            classification = 1.0
        elif trade_price < mid:
            classification = -1.0
        elif self._has_prev:
            # Tick rule: compare to previous trade
            if trade_price > self._prev_trade_price:
                classification = 1.0
            elif trade_price < self._prev_trade_price:
                classification = -1.0
            else:
                classification = 0.0  # indeterminate
        else:
            classification = 0.0

        self._prev_trade_price = trade_price
        self._has_prev = True
        return classification

    def reset(self) -> None:
        self._prev_trade_price = 0.0
        self._has_prev = False


def classify_trade(trade_price: float, bid: float, ask: float, prev_trade: float | None) -> int:
    """Standalone classification function for use by other features (e.g., VPIN).

    Returns +1 (buyer), -1 (seller), or 0 (unknown).
    """
    mid = (bid + ask) / 2.0
    if trade_price > mid:
        return 1
    elif trade_price < mid:
        return -1
    elif prev_trade is not None:
        if trade_price > prev_trade:
            return 1
        elif trade_price < prev_trade:
            return -1
    return 0
