"""Tests for adaptive stop mechanics."""
from __future__ import annotations

import pytest
from src.backtest.position_manager import AdaptiveStopConfig


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
