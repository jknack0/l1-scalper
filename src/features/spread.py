"""Bid-ask spread in ticks."""

from __future__ import annotations

from src.data.schemas import L1Record
from src.features.base import Feature

MES_TICK_SIZE = 0.25


class Spread(Feature):
    """Bid-ask spread in ticks. MES tick = 0.25 points."""

    name = "spread"

    def update(self, record: L1Record) -> float | None:
        spread_points = record.ask_price - record.bid_price
        return spread_points / MES_TICK_SIZE

    def reset(self) -> None:
        pass
