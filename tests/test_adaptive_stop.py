"""Tests for adaptive stop mechanics."""
from __future__ import annotations

import numpy as np
import pytest
from src.backtest.position_manager import (
    AdaptiveStopConfig, PositionManager, Side, Trade, MES_TICK,
)


class TestAdaptiveStopConfig:
    def test_defaults_all_disabled(self):
        """Default config is just hard SL, everything else off."""
        cfg = AdaptiveStopConfig()
        assert cfg.hard_sl_ticks == 8.0
        assert cfg.breakeven_trigger_ticks == 0.0
        assert cfg.tier1_activation_ticks == 0.0
        assert cfg.velocity_lookback_bars == 0

    def test_valid_tiered_config(self):
        """Valid 3-tier config passes validation."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=3.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=6.0, tier2_trail_distance=1.5,
            tier3_activation_ticks=10.0, tier3_trail_distance=0.5,
        )
        cfg.validate()

    def test_tier_activation_order_violated(self):
        """Tier activations must be strictly increasing."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=6.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=4.0, tier2_trail_distance=1.5,
        )
        with pytest.raises(ValueError, match="tier2_activation.*tier1"):
            cfg.validate()

    def test_tier_distance_order_violated(self):
        """Tier distances must be strictly decreasing."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=3.0, tier1_trail_distance=1.0,
            tier2_activation_ticks=6.0, tier2_trail_distance=2.0,
        )
        with pytest.raises(ValueError, match="tier2_trail.*tier1"):
            cfg.validate()

    def test_disabled_tier_disables_higher(self):
        """If tier2 disabled, tier3 must also be disabled."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=3.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=0.0,
            tier3_activation_ticks=10.0, tier3_trail_distance=0.5,
        )
        with pytest.raises(ValueError, match="tier3.*tier2.*disabled"):
            cfg.validate()

    def test_all_disabled_valid(self):
        """All optional mechanics disabled is valid (hard SL only)."""
        cfg = AdaptiveStopConfig(hard_sl_ticks=6.0)
        cfg.validate()


class TestAdaptiveStopMechanics:
    """Test the adaptive stop exit logic via PositionManager."""

    def _run_position(
        self, config: AdaptiveStopConfig, mid_prices: list[float], p_up: float = 0.80,
    ) -> list[Trade]:
        """Helper: open a long at bar 0 via high P(up), feed mid prices."""
        pm = PositionManager(config)
        pm.update(0, p_up, mid_prices[0])
        assert pm.position_side == Side.LONG
        for i in range(1, len(mid_prices)):
            pm.update(i, 0.5, mid_prices[i])
        return pm.trades

    def test_hard_sl_only(self):
        cfg = AdaptiveStopConfig(hard_sl_ticks=4.0)
        cfg.validate()
        mids = [5000.00] + [5000.00 - i * MES_TICK for i in range(1, 6)]
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "hard_sl"

    def test_breakeven_lock(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            breakeven_trigger_ticks=3.0,
            breakeven_lock_ticks=1.0,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        for i in range(1, 5):
            mids.append(entry + i * MES_TICK)
        for i in range(4, -5, -1):
            mids.append(entry + i * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "breakeven"

    def test_tier1_trail(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            tier1_activation_ticks=4.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        for i in range(1, 7):
            mids.append(entry + i * MES_TICK)
        mids.append(entry + 5 * MES_TICK)
        mids.append(entry + 4 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].pnl_ticks == pytest.approx(4.0)

    def test_tier_escalation(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            tier1_activation_ticks=3.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=6.0, tier2_trail_distance=1.0,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        for i in range(1, 9):
            mids.append(entry + i * MES_TICK)
        mids.append(entry + 7 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier2"
        assert trades[0].pnl_ticks == pytest.approx(7.0)

    def test_velocity_ratchet(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            tier1_activation_ticks=2.0, tier1_trail_distance=4.0,
            velocity_lookback_bars=3,
            velocity_threshold_ticks=4.0,
            velocity_trail_distance=0.5,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        for i in range(1, 4):
            mids.append(entry + i * MES_TICK)
        mids.append(entry + 5 * MES_TICK)
        mids.append(entry + 6 * MES_TICK)
        mids.append(entry + 7 * MES_TICK)
        mids.append(entry + 6 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "velocity"

    def test_cross_bar_ratchet_never_loosens(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            velocity_lookback_bars=2,
            velocity_threshold_ticks=3.0,
            velocity_trail_distance=1.0,
            tier1_activation_ticks=2.0,
            tier1_trail_distance=4.0,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        mids.append(entry + 2 * MES_TICK)
        mids.append(entry + 4 * MES_TICK)
        mids.append(entry + 4 * MES_TICK)
        mids.append(entry + 3 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].pnl_ticks == pytest.approx(3.0)

    def test_breakeven_and_tier_interaction(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            breakeven_trigger_ticks=3.0,
            breakeven_lock_ticks=1.5,
            tier1_activation_ticks=2.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        entry = 5000.00
        mids = [entry]
        for i in range(1, 6):
            mids.append(entry + i * MES_TICK)
        mids.append(entry + 4 * MES_TICK)
        mids.append(entry + 3 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].pnl_ticks == pytest.approx(3.0)

    def test_max_hold_still_works(self):
        cfg = AdaptiveStopConfig(hard_sl_ticks=100.0, max_hold_bars=5)
        cfg.validate()
        mids = [5000.00] * 7
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "max_hold"

    def test_short_position(self):
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            tier1_activation_ticks=3.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        pm = PositionManager(cfg)
        entry = 5000.00
        pm.update(0, 0.15, entry)
        assert pm.position_side == Side.SHORT
        for i in range(1, 6):
            pm.update(i, 0.5, entry - i * MES_TICK)
        pm.update(6, 0.5, entry - 4 * MES_TICK)
        pm.update(7, 0.5, entry - 3 * MES_TICK)
        trades = pm.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].side == Side.SHORT
        assert trades[0].pnl_ticks == pytest.approx(3.0)


from src.backtest.engine import run_backtest, BacktestResult


class TestBacktestEngineAdaptive:
    def test_run_backtest_with_adaptive_config(self):
        """Engine accepts AdaptiveStopConfig and reports adaptive exit reasons."""
        n = 200
        mid = np.full(n, 5000.0)
        p_up = np.full(n, 0.5)
        p_up[10] = 0.80

        for i in range(11, 20):
            mid[i] = 5000.0 + min(i - 10, 6) * MES_TICK
        for i in range(20, 30):
            mid[i] = 5000.0 + max(6 - (i - 19), 0) * MES_TICK

        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            tier1_activation_ticks=4.0,
            tier1_trail_distance=2.0,
        )
        result = run_backtest(p_up, mid, config=cfg)
        assert result.n_trades >= 1
        assert result.exits_tier1 >= 1
        assert result.exits_hard_sl == 0

    def test_backward_compat_old_config(self):
        """Engine still works with old PositionManagerConfig."""
        from src.backtest.position_manager import PositionManagerConfig
        n = 50
        mid = np.full(n, 5000.0)
        p_up = np.full(n, 0.5)
        p_up[5] = 0.80
        mid[10:] = 5000.0 - 15 * MES_TICK
        cfg = PositionManagerConfig(hard_sl_ticks=8.0)
        result = run_backtest(p_up, mid, config=cfg)
        assert result.n_trades >= 1
