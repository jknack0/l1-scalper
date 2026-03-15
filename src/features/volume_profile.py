"""Volume Profile: POC, VAH, VAL from session trade distribution."""

from __future__ import annotations

from collections import defaultdict
from math import isnan

from src.data.schemas import L1Record
from src.features.base import Feature

MES_TICK_SIZE = 0.25
VALUE_AREA_PCT = 0.70


class VolumeProfile(Feature):
    """Session volume profile with 3 outputs: dist to POC, VAH, VAL in ticks.

    Returns distance from current price to POC (primary output).
    Use get_levels() for all three values.
    """

    name = "volume_profile"

    def __init__(self) -> None:
        self._volume_at_price: defaultdict[float, int] = defaultdict(int)
        self._total_volume: int = 0
        self._poc: float = 0.0
        self._vah: float = 0.0
        self._val: float = 0.0
        self._last_price: float = 0.0

    def update(self, record: L1Record) -> float | None:
        mid = (record.bid_price + record.ask_price) / 2.0
        self._last_price = mid

        if record.event_type == "trade" and not isnan(record.trade_price):
            # Round to tick
            price = round(record.trade_price / MES_TICK_SIZE) * MES_TICK_SIZE
            self._volume_at_price[price] += record.trade_size
            self._total_volume += record.trade_size
            self._recompute_levels()

        if self._total_volume == 0:
            return None

        return (mid - self._poc) / MES_TICK_SIZE

    def get_levels(self) -> tuple[float, float, float]:
        """Return (dist_to_poc, dist_to_vah, dist_to_val) in ticks."""
        p = self._last_price
        return (
            (p - self._poc) / MES_TICK_SIZE,
            (p - self._vah) / MES_TICK_SIZE,
            (p - self._val) / MES_TICK_SIZE,
        )

    def _recompute_levels(self) -> None:
        if not self._volume_at_price:
            return

        # POC = price with highest volume
        self._poc = max(self._volume_at_price, key=self._volume_at_price.get)

        # Value area: 70% of volume centered on POC
        target_vol = int(self._total_volume * VALUE_AREA_PCT)
        sorted_prices = sorted(self._volume_at_price.keys())

        if not sorted_prices:
            return

        poc_idx = sorted_prices.index(self._poc)
        lo = poc_idx
        hi = poc_idx
        area_vol = self._volume_at_price[self._poc]

        while area_vol < target_vol and (lo > 0 or hi < len(sorted_prices) - 1):
            expand_lo = self._volume_at_price[sorted_prices[lo - 1]] if lo > 0 else -1
            expand_hi = self._volume_at_price[sorted_prices[hi + 1]] if hi < len(sorted_prices) - 1 else -1

            if expand_lo >= expand_hi and lo > 0:
                lo -= 1
                area_vol += self._volume_at_price[sorted_prices[lo]]
            elif hi < len(sorted_prices) - 1:
                hi += 1
                area_vol += self._volume_at_price[sorted_prices[hi]]
            else:
                break

        self._val = sorted_prices[lo]
        self._vah = sorted_prices[hi]

    def reset(self) -> None:
        self._volume_at_price.clear()
        self._total_volume = 0
        self._poc = 0.0
        self._vah = 0.0
        self._val = 0.0
        self._last_price = 0.0
