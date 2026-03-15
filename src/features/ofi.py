"""Order Flow Imbalance (OFI) from consecutive BBO snapshots.

Implements the Cont et al. (2014) definition using L1 best bid/ask only.
"""

from __future__ import annotations

from src.data.schemas import L1Record
from src.features.base import Feature


class OrderFlowImbalance(Feature):
    """OFI computed from consecutive BBO quote updates.

    delta_bid = (bid_up) * bid_size_now - (bid_down) * bid_size_prev
                + (bid_same) * (bid_size_now - bid_size_prev)
    delta_ask = similar for ask side
    OFI = delta_bid - delta_ask
    """

    name = "ofi"

    def __init__(self) -> None:
        self._prev_bid_price: float = 0.0
        self._prev_bid_size: int = 0
        self._prev_ask_price: float = 0.0
        self._prev_ask_size: int = 0
        self._has_prev: bool = False

    def update(self, record: L1Record) -> float | None:
        bid_px = record.bid_price
        bid_sz = record.bid_size
        ask_px = record.ask_price
        ask_sz = record.ask_size

        if not self._has_prev:
            self._prev_bid_price = bid_px
            self._prev_bid_size = bid_sz
            self._prev_ask_price = ask_px
            self._prev_ask_size = ask_sz
            self._has_prev = True
            return None

        # Bid side delta
        if bid_px > self._prev_bid_price:
            delta_bid = bid_sz
        elif bid_px < self._prev_bid_price:
            delta_bid = -self._prev_bid_size
        else:
            delta_bid = bid_sz - self._prev_bid_size

        # Ask side delta
        if ask_px > self._prev_ask_price:
            delta_ask = -self._prev_ask_size
        elif ask_px < self._prev_ask_price:
            delta_ask = ask_sz
        else:
            delta_ask = ask_sz - self._prev_ask_size

        ofi = float(delta_bid - delta_ask)

        self._prev_bid_price = bid_px
        self._prev_bid_size = bid_sz
        self._prev_ask_price = ask_px
        self._prev_ask_size = ask_sz

        return ofi

    def reset(self) -> None:
        self._prev_bid_price = 0.0
        self._prev_bid_size = 0
        self._prev_ask_price = 0.0
        self._prev_ask_size = 0
        self._has_prev = False
