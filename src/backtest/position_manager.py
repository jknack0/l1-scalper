"""Position manager — state machine driven by rolling P(up) signal.

Manages a single MES position based on entry/exit thresholds applied
to the continuous P(up) output from the rolling inference engine.

States: FLAT → LONG or SHORT → FLAT
Transitions driven by P(up) crossing thresholds.

Safety: hard SL (ticks from entry) and max hold time as circuit breakers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

MES_TICK = 0.25
MES_TICK_VALUE = 1.25  # $1.25 per tick


class Side(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Completed trade record."""
    entry_bar: int
    exit_bar: int
    side: Side
    entry_price: float       # mid at entry
    exit_price: float        # mid at exit
    entry_p_up: float        # P(up) at entry
    exit_p_up: float         # P(up) at exit
    exit_reason: str         # "signal", "hard_sl", "max_hold", "session_end"
    pnl_ticks: float         # realized P&L in ticks
    hold_bars: int           # duration in seconds


@dataclass
class PositionManagerConfig:
    """Thresholds and safety parameters."""
    # Entry thresholds
    long_entry: float = 0.70    # P(up) >= this → enter long
    short_entry: float = 0.30   # P(up) <= this → enter short

    # Exit thresholds (signal-based)
    long_exit: float = 0.50     # P(up) drops below this → exit long
    short_exit: float = 0.50    # P(up) rises above this → exit short

    # Safety circuit breakers
    hard_sl_ticks: float = 12.0     # max adverse move before forced exit
    max_hold_bars: int = 300        # 5 minutes max hold

    # Trailing stop (matches live bot logic)
    trail_activation_ticks: float = 0.0   # runup needed to activate trail (0 = disabled)
    trail_distance_ticks: float = 0.0     # drawback from best to trigger trail exit

    # Commission
    commission_rt_dollars: float = 0.59  # round-trip commission


@dataclass
class AdaptiveStopConfig:
    """Unified adaptive stop configuration. All params sweepable."""
    # Entry thresholds (carried over from PositionManagerConfig)
    long_entry: float = 0.70
    short_entry: float = 0.30
    max_hold_bars: int = 300
    commission_rt_dollars: float = 0.59

    # Hard stop
    hard_sl_ticks: float = 8.0

    # Breakeven lock
    breakeven_trigger_ticks: float = 0.0
    breakeven_lock_ticks: float = 0.0

    # 3-Tier trailing stop
    tier1_activation_ticks: float = 0.0
    tier1_trail_distance: float = 2.0
    tier2_activation_ticks: float = 0.0
    tier2_trail_distance: float = 1.0
    tier3_activation_ticks: float = 0.0
    tier3_trail_distance: float = 0.5

    # Velocity ratchet
    velocity_lookback_bars: int = 0
    velocity_threshold_ticks: float = 0.0
    velocity_trail_distance: float = 0.5

    def validate(self) -> None:
        """Raise ValueError if constraints violated."""
        active_tiers = []
        for i, act in enumerate([
            self.tier1_activation_ticks,
            self.tier2_activation_ticks,
            self.tier3_activation_ticks,
        ], 1):
            if act > 0:
                active_tiers.append((i, act))

        for j in range(1, len(active_tiers)):
            prev_idx, prev_act = active_tiers[j - 1]
            cur_idx, cur_act = active_tiers[j]
            if cur_act <= prev_act:
                raise ValueError(
                    f"tier{cur_idx}_activation ({cur_act}) must be > "
                    f"tier{prev_idx}_activation ({prev_act})"
                )

        tiers_enabled = [
            self.tier1_activation_ticks > 0,
            self.tier2_activation_ticks > 0,
            self.tier3_activation_ticks > 0,
        ]
        for i in range(1, 3):
            if tiers_enabled[i] and not tiers_enabled[i - 1]:
                raise ValueError(
                    f"tier{i + 1} enabled but tier{i} disabled"
                )

        dists = [
            (1, self.tier1_activation_ticks, self.tier1_trail_distance),
            (2, self.tier2_activation_ticks, self.tier2_trail_distance),
            (3, self.tier3_activation_ticks, self.tier3_trail_distance),
        ]
        active_dists = [(idx, d) for idx, act, d in dists if act > 0]
        for j in range(1, len(active_dists)):
            prev_idx, prev_d = active_dists[j - 1]
            cur_idx, cur_d = active_dists[j]
            if cur_d >= prev_d:
                raise ValueError(
                    f"tier{cur_idx}_trail ({cur_d}) must be < "
                    f"tier{prev_idx}_trail ({prev_d})"
                )


class PositionManager:
    """State machine that converts P(up) stream into trade decisions.

    Call `update()` every bar with the current P(up) and mid price.
    Completed trades are appended to `self.trades`.
    """

    def __init__(self, config: PositionManagerConfig | None = None) -> None:
        self.config = config or PositionManagerConfig()
        self.trades: list[Trade] = []

        # Current position state
        self._side = Side.FLAT
        self._entry_bar: int = 0
        self._entry_price: float = 0.0
        self._entry_p_up: float = 0.0
        self._best_price: float = 0.0
        self._trail_active: bool = False

    def update(self, bar_idx: int, p_up: float, mid: float) -> Trade | None:
        """Process one bar. Returns a Trade if a position was closed this bar.

        Args:
            bar_idx: current bar index (for timestamps).
            p_up: P(up) from rolling inference. NaN = no signal (warmup).
            mid: mid price at this bar.

        Returns:
            Completed Trade if a position was closed, else None.
        """
        if np.isnan(p_up):
            return None

        cfg = self.config

        # If we have a position, check exits first
        if self._side != Side.FLAT:
            return self._check_exit(bar_idx, p_up, mid)

        # Flat — check for entry
        if p_up >= cfg.long_entry:
            self._open(bar_idx, p_up, mid, Side.LONG)
        elif p_up <= cfg.short_entry:
            self._open(bar_idx, p_up, mid, Side.SHORT)

        return None

    def _open(self, bar_idx: int, p_up: float, mid: float, side: Side) -> None:
        self._side = side
        self._entry_bar = bar_idx
        self._entry_price = mid
        self._entry_p_up = p_up
        self._best_price = mid
        self._trail_active = False

    def _close(self, bar_idx: int, p_up: float, mid: float, reason: str) -> Trade:
        if self._side == Side.LONG:
            pnl_ticks = (mid - self._entry_price) / MES_TICK
        else:
            pnl_ticks = (self._entry_price - mid) / MES_TICK

        trade = Trade(
            entry_bar=self._entry_bar,
            exit_bar=bar_idx,
            side=self._side,
            entry_price=self._entry_price,
            exit_price=mid,
            entry_p_up=self._entry_p_up,
            exit_p_up=p_up,
            exit_reason=reason,
            pnl_ticks=pnl_ticks,
            hold_bars=bar_idx - self._entry_bar,
        )
        self.trades.append(trade)
        self._side = Side.FLAT
        return trade

    def _check_exit(self, bar_idx: int, p_up: float, mid: float) -> Trade | None:
        cfg = self.config
        hold_bars = bar_idx - self._entry_bar

        # Track best price (MFE)
        if self._side == Side.LONG:
            self._best_price = max(self._best_price, mid)
            runup = (self._best_price - self._entry_price) / MES_TICK
            current_pnl = (mid - self._entry_price) / MES_TICK
        else:
            self._best_price = min(self._best_price, mid)
            runup = (self._entry_price - self._best_price) / MES_TICK
            current_pnl = (self._entry_price - mid) / MES_TICK

        # 1. Hard SL check
        if current_pnl <= -cfg.hard_sl_ticks:
            return self._close(bar_idx, p_up, mid, "hard_sl")

        # 2. Trailing stop (matches live bot logic)
        if cfg.trail_activation_ticks > 0:
            if not self._trail_active and runup >= cfg.trail_activation_ticks:
                self._trail_active = True
            if self._trail_active:
                drawback = runup - current_pnl
                if drawback >= cfg.trail_distance_ticks:
                    return self._close(bar_idx, p_up, mid, "trail")

        # 3. Max hold time
        if hold_bars >= cfg.max_hold_bars:
            return self._close(bar_idx, p_up, mid, "max_hold")

        # 4. Signal-based exit (only if trail not active — trail takes priority)
        if not self._trail_active:
            if self._side == Side.LONG and p_up < cfg.long_exit:
                return self._close(bar_idx, p_up, mid, "signal")
            if self._side == Side.SHORT and p_up > cfg.short_exit:
                return self._close(bar_idx, p_up, mid, "signal")

        return None

    def force_close(self, bar_idx: int, p_up: float, mid: float) -> Trade | None:
        """Force close any open position (e.g., session end)."""
        if self._side != Side.FLAT:
            return self._close(bar_idx, p_up, mid, "session_end")
        return None

    @property
    def is_flat(self) -> bool:
        return self._side == Side.FLAT

    @property
    def position_side(self) -> Side:
        return self._side

    def reset(self) -> None:
        """Reset state for a new session (does NOT clear trade history)."""
        self._side = Side.FLAT
        self._trail_active = False
