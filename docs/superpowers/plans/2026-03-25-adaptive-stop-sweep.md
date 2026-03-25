# Adaptive Stop Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed trailing stop with a unified adaptive stop system (hard SL + breakeven lock + 3-tier trail + velocity ratchet) and sweep all parameters per regime pair using random search in a walk-forward framework.

**Architecture:** New `AdaptiveStopConfig` dataclass and rewritten `PositionManager._check_exit()` that computes a composite stop level each bar. New `scripts/sweep_adaptive_stop.py` that random-samples configs and evaluates per regime pair with walk-forward validation. The existing `run_backtest()` is updated to accept the new config and report exit sub-reasons.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, Click, Pandas, PyArrow

**Spec:** `docs/superpowers/specs/2026-03-25-adaptive-stop-sweep-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/backtest/position_manager.py` | Modify | Add `AdaptiveStopConfig`, rewrite `_check_exit()`, add ring buffer for velocity |
| `src/backtest/engine.py` | Modify | Update `BacktestResult` exit reason fields, pass new config type |
| `scripts/sweep_adaptive_stop.py` | Create | Random search sweep with walk-forward, robustness checks, output files |
| `tests/test_adaptive_stop.py` | Create | Unit tests for adaptive stop mechanics |

---

### Task 1: AdaptiveStopConfig dataclass and validation

**Files:**
- Modify: `src/backtest/position_manager.py:32-68`
- Test: `tests/test_adaptive_stop.py`

- [ ] **Step 1: Write failing tests for config validation**

```python
# tests/test_adaptive_stop.py
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
        cfg.validate()  # should not raise

    def test_tier_activation_order_violated(self):
        """Tier activations must be strictly increasing."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=6.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=4.0, tier2_trail_distance=1.5,  # < tier1
        )
        with pytest.raises(ValueError, match="tier2_activation.*tier1"):
            cfg.validate()

    def test_tier_distance_order_violated(self):
        """Tier distances must be strictly decreasing."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=3.0, tier1_trail_distance=1.0,
            tier2_activation_ticks=6.0, tier2_trail_distance=2.0,  # > tier1
        )
        with pytest.raises(ValueError, match="tier2_trail.*tier1"):
            cfg.validate()

    def test_disabled_tier_disables_higher(self):
        """If tier2 disabled, tier3 must also be disabled."""
        cfg = AdaptiveStopConfig(
            tier1_activation_ticks=3.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=0.0,  # disabled
            tier3_activation_ticks=10.0, tier3_trail_distance=0.5,  # enabled!
        )
        with pytest.raises(ValueError, match="tier3.*tier2.*disabled"):
            cfg.validate()

    def test_all_disabled_valid(self):
        """All optional mechanics disabled is valid (hard SL only)."""
        cfg = AdaptiveStopConfig(hard_sl_ticks=6.0)
        cfg.validate()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_adaptive_stop.py -v`
Expected: FAIL (AdaptiveStopConfig not defined)

- [ ] **Step 3: Implement AdaptiveStopConfig**

Add to `src/backtest/position_manager.py` after the `PositionManagerConfig` class:

```python
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
    breakeven_trigger_ticks: float = 0.0   # 0 = disabled
    breakeven_lock_ticks: float = 0.0

    # 3-Tier trailing stop
    tier1_activation_ticks: float = 0.0    # 0 = no trail
    tier1_trail_distance: float = 2.0
    tier2_activation_ticks: float = 0.0    # 0 = skip
    tier2_trail_distance: float = 1.0
    tier3_activation_ticks: float = 0.0    # 0 = skip
    tier3_trail_distance: float = 0.5

    # Velocity ratchet
    velocity_lookback_bars: int = 0        # 0 = disabled
    velocity_threshold_ticks: float = 0.0
    velocity_trail_distance: float = 0.5

    def validate(self) -> None:
        """Raise ValueError if constraints violated."""
        # Tier activation order
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

        # Disabled tier disables higher tiers
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

        # Tier distance order (only for active tiers)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_adaptive_stop.py::TestAdaptiveStopConfig -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/position_manager.py tests/test_adaptive_stop.py
git commit -m "feat: add AdaptiveStopConfig with validation"
```

---

### Task 2: Adaptive stop exit logic in PositionManager

**Files:**
- Modify: `src/backtest/position_manager.py:70-205`
- Test: `tests/test_adaptive_stop.py`

- [ ] **Step 1: Write failing tests for stop mechanics**

Append to `tests/test_adaptive_stop.py`:

```python
import numpy as np
from src.backtest.position_manager import (
    AdaptiveStopConfig, PositionManager, Side, Trade, MES_TICK,
)


class TestAdaptiveStopMechanics:
    """Test the adaptive stop exit logic via PositionManager."""

    def _run_position(
        self, config: AdaptiveStopConfig, mid_prices: list[float], p_up: float = 0.80,
    ) -> list[Trade]:
        """Helper: open a long at bar 0 via high P(up), feed mid prices."""
        pm = PositionManager(config)
        # Bar 0: entry
        pm.update(0, p_up, mid_prices[0])
        assert pm.position_side == Side.LONG
        # Subsequent bars: feed with neutral P(up) so no re-entry
        for i in range(1, len(mid_prices)):
            pm.update(i, 0.5, mid_prices[i])
        return pm.trades

    def test_hard_sl_only(self):
        """With all optional mechanics off, hard SL triggers on adverse move."""
        cfg = AdaptiveStopConfig(hard_sl_ticks=4.0)
        cfg.validate()
        # Entry at 5000.00, then price drops 4 ticks = 1.00
        mids = [5000.00] + [5000.00 - i * MES_TICK for i in range(1, 6)]
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "hard_sl"

    def test_breakeven_lock(self):
        """Breakeven lock moves stop to entry + lock after trigger hit."""
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            breakeven_trigger_ticks=3.0,
            breakeven_lock_ticks=1.0,
        )
        cfg.validate()
        # Go up 3 ticks (trigger), then drop back below lock level
        entry = 5000.00
        mids = [entry]
        # Rise 4 ticks
        for i in range(1, 5):
            mids.append(entry + i * MES_TICK)
        # Drop back: should exit when pnl <= 1.0 tick (lock level)
        for i in range(4, -5, -1):
            mids.append(entry + i * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "breakeven"

    def test_tier1_trail(self):
        """Tier 1 activates at threshold and trails from MFE."""
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            tier1_activation_ticks=4.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        entry = 5000.00
        # Rise 6 ticks (activates tier1 at 4), then drop 2 from MFE
        mids = [entry]
        for i in range(1, 7):
            mids.append(entry + i * MES_TICK)
        # MFE = 6 ticks, trail distance = 2, so stop at 4 ticks profit
        # Drop: 5, 4 ticks profit -> exit at 4
        mids.append(entry + 5 * MES_TICK)
        mids.append(entry + 4 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].pnl_ticks == pytest.approx(4.0)

    def test_tier_escalation(self):
        """Higher tiers tighten the trail as price advances."""
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            tier1_activation_ticks=3.0, tier1_trail_distance=3.0,
            tier2_activation_ticks=6.0, tier2_trail_distance=1.0,
        )
        cfg.validate()
        entry = 5000.00
        # Rise 8 ticks (tier2 active at 6)
        mids = [entry]
        for i in range(1, 9):
            mids.append(entry + i * MES_TICK)
        # MFE = 8, tier2 trail = 1.0, stop at 7.0
        # Drop to 7 ticks -> exit
        mids.append(entry + 7 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier2"
        assert trades[0].pnl_ticks == pytest.approx(7.0)

    def test_velocity_ratchet(self):
        """Velocity ratchet tightens stop when price moves fast."""
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
        # Slow rise: 1 tick per bar for 3 bars
        for i in range(1, 4):
            mids.append(entry + i * MES_TICK)
        # Fast rip: +4 ticks in 3 bars (bars 4, 5, 6)
        mids.append(entry + 5 * MES_TICK)   # bar 4: +2 from bar 3
        mids.append(entry + 6 * MES_TICK)   # bar 5
        mids.append(entry + 7 * MES_TICK)   # bar 6: moved +4 from bar 3
        # Velocity triggered: MFE=7, velocity trail=0.5, stop at 6.5
        # Drop to 6.5 -> exit
        mids.append(entry + 6 * MES_TICK)   # bar 7: pnl=6, stop was 6.5 -> exit
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "velocity"

    def test_cross_bar_ratchet_never_loosens(self):
        """Stop level must never decrease across bars."""
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
        # Rise fast: +4 ticks in 2 bars (velocity triggers)
        mids.append(entry + 2 * MES_TICK)  # bar 1
        mids.append(entry + 4 * MES_TICK)  # bar 2: velocity fires, MFE=4, stop=3.0
        # Stall: velocity no longer fires, but stop must NOT drop below 3.0
        mids.append(entry + 4 * MES_TICK)  # bar 3: no velocity, tier1 stop = 4-4=0
        # Without cross-bar ratchet, stop would drop to max(0, -20) = 0
        # With ratchet, stop stays at 3.0
        # Drop to 3.0 -> should exit
        mids.append(entry + 3 * MES_TICK)  # bar 4: pnl=3.0, stop=3.0 -> exit
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].pnl_ticks == pytest.approx(3.0)

    def test_breakeven_and_tier_interaction(self):
        """When both breakeven and tier1 are active, tightest wins."""
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=20.0,
            breakeven_trigger_ticks=3.0,
            breakeven_lock_ticks=1.5,
            tier1_activation_ticks=2.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        entry = 5000.00
        # Rise to MFE=5: breakeven stop=1.5, tier1 stop=5-2=3.0
        # Tier1 is tighter (3.0 > 1.5), so tier1 binds
        mids = [entry]
        for i in range(1, 6):
            mids.append(entry + i * MES_TICK)
        # Drop to 3.0 ticks profit -> exit via tier1 (not breakeven)
        mids.append(entry + 4 * MES_TICK)
        mids.append(entry + 3 * MES_TICK)
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].pnl_ticks == pytest.approx(3.0)

    def test_max_hold_still_works(self):
        """Max hold fires even if no stop mechanic triggers."""
        cfg = AdaptiveStopConfig(hard_sl_ticks=100.0, max_hold_bars=5)
        cfg.validate()
        mids = [5000.00] * 7
        trades = self._run_position(cfg, mids)
        assert len(trades) == 1
        assert trades[0].exit_reason == "max_hold"

    def test_short_position(self):
        """Adaptive stop works symmetrically for short positions."""
        cfg = AdaptiveStopConfig(
            hard_sl_ticks=10.0,
            tier1_activation_ticks=3.0,
            tier1_trail_distance=2.0,
        )
        cfg.validate()
        pm = PositionManager(cfg)
        entry = 5000.00
        # Enter short
        pm.update(0, 0.15, entry)
        assert pm.position_side == Side.SHORT
        # Price drops (favorable for short) 5 ticks
        for i in range(1, 6):
            pm.update(i, 0.5, entry - i * MES_TICK)
        # MFE = 5 ticks, trail distance = 2, stop at 3 ticks profit
        # Price rises back (unfavorable): pnl drops from 5 to 3 -> exit
        pm.update(6, 0.5, entry - 4 * MES_TICK)
        pm.update(7, 0.5, entry - 3 * MES_TICK)
        trades = pm.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "tier1"
        assert trades[0].side == Side.SHORT
        assert trades[0].pnl_ticks == pytest.approx(3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_adaptive_stop.py::TestAdaptiveStopMechanics -v`
Expected: FAIL (PositionManager doesn't accept AdaptiveStopConfig)

- [ ] **Step 3: Rewrite PositionManager to support AdaptiveStopConfig**

Modify `src/backtest/position_manager.py`:

1. Update `PositionManager.__init__` to accept `AdaptiveStopConfig | PositionManagerConfig | None`:
   - If `AdaptiveStopConfig`, use new adaptive exit logic
   - If `PositionManagerConfig` (backward compat), use old logic
   - Store `self._use_adaptive: bool`

2. Add state fields in `__init__`:
   - `self._mfe: float = 0.0`
   - `self._prev_stop_level: float`
   - `self._pnl_history: list[float]` (ring buffer via list, reset per trade)

3. Update `_open()` to reset adaptive state:
   - `self._mfe = 0.0`
   - `self._prev_stop_level = -cfg.hard_sl_ticks`
   - `self._pnl_history = []`

4. Add `_check_exit_adaptive()` implementing the spec pseudocode. **IMPORTANT: This method must NOT include signal-based exits (no `long_exit` / `short_exit` threshold checks). `AdaptiveStopConfig` intentionally omits these fields. Only adaptive stop mechanics + max_hold apply.**

   ```python
   def _check_exit_adaptive(self, bar_idx: int, p_up: float, mid: float) -> Trade | None:
       cfg = self.config
       hold_bars = bar_idx - self._entry_bar

       # Compute current P&L in ticks
       if self._side == Side.LONG:
           current_pnl = (mid - self._entry_price) / MES_TICK
       else:
           current_pnl = (self._entry_price - mid) / MES_TICK

       self._mfe = max(self._mfe, current_pnl)
       self._pnl_history.append(current_pnl)

       # Track binding mechanic alongside stop_level
       stop_level = -cfg.hard_sl_ticks
       binding = "hard_sl"

       # Breakeven lock
       if cfg.breakeven_trigger_ticks > 0 and self._mfe >= cfg.breakeven_trigger_ticks:
           be_level = cfg.breakeven_lock_ticks
           if be_level > stop_level:
               stop_level = be_level
               binding = "breakeven"

       # Tiered trail (highest active tier wins, uses mfe for activation)
       if cfg.tier3_activation_ticks > 0 and self._mfe >= cfg.tier3_activation_ticks:
           t_level = self._mfe - cfg.tier3_trail_distance
           if t_level > stop_level:
               stop_level = t_level
               binding = "tier3"
       elif cfg.tier2_activation_ticks > 0 and self._mfe >= cfg.tier2_activation_ticks:
           t_level = self._mfe - cfg.tier2_trail_distance
           if t_level > stop_level:
               stop_level = t_level
               binding = "tier2"
       elif cfg.tier1_activation_ticks > 0 and self._mfe >= cfg.tier1_activation_ticks:
           t_level = self._mfe - cfg.tier1_trail_distance
           if t_level > stop_level:
               stop_level = t_level
               binding = "tier1"

       # Velocity ratchet
       if (cfg.velocity_lookback_bars > 0
               and len(self._pnl_history) >= cfg.velocity_lookback_bars):
           recent_move = current_pnl - self._pnl_history[-cfg.velocity_lookback_bars]
           if recent_move >= cfg.velocity_threshold_ticks:
               v_level = self._mfe - cfg.velocity_trail_distance
               if v_level > stop_level:
                   stop_level = v_level
                   binding = "velocity"

       # Cross-bar ratchet: stop never moves backward
       if stop_level > self._prev_stop_level:
           self._prev_stop_level = stop_level
       else:
           stop_level = self._prev_stop_level
           # binding stays from whichever bar set the higher level

       # Max hold (takes priority as safety)
       if hold_bars >= cfg.max_hold_bars:
           return self._close(bar_idx, p_up, mid, "max_hold")

       # Exit check
       if current_pnl <= stop_level:
           return self._close(bar_idx, p_up, mid, binding)

       return None
   ```

5. Update `_check_exit()` to dispatch to `_check_exit_adaptive()` when `self._use_adaptive`.

6. Update `Trade.exit_reason` doc comment to include new reasons: `"hard_sl"`, `"breakeven"`, `"tier1"`, `"tier2"`, `"tier3"`, `"velocity"`, `"max_hold"`, `"session_end"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_adaptive_stop.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/position_manager.py tests/test_adaptive_stop.py
git commit -m "feat: adaptive stop exit logic with 4 mechanics + cross-bar ratchet"
```

---

### Task 3: Update BacktestResult and engine for adaptive stop

**Files:**
- Modify: `src/backtest/engine.py:36-60, 119, 157-182`
- Test: `tests/test_adaptive_stop.py`

- [ ] **Step 1: Write failing test for engine with adaptive config**

Append to `tests/test_adaptive_stop.py`:

```python
from src.backtest.engine import run_backtest, BacktestResult


class TestBacktestEngineAdaptive:
    def test_run_backtest_with_adaptive_config(self):
        """Engine accepts AdaptiveStopConfig and reports adaptive exit reasons."""
        n = 200
        mid = np.full(n, 5000.0)
        # Simulate: signal fires at bar 10, price rips 6 ticks, retraces
        p_up = np.full(n, 0.5)
        p_up[10] = 0.80  # long entry

        # Price action: rise 6 ticks from bar 10, then drop
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
        mid[10:] = 5000.0 - 15 * MES_TICK  # big drop -> SL
        cfg = PositionManagerConfig(hard_sl_ticks=8.0)
        result = run_backtest(p_up, mid, config=cfg)
        assert result.n_trades >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_adaptive_stop.py::TestBacktestEngineAdaptive -v`
Expected: FAIL (BacktestResult missing exits_tier1, etc.)

- [ ] **Step 3: Update BacktestResult and engine**

Modify `src/backtest/engine.py`:

1. Update `BacktestResult` dataclass — replace `exits_trail` with adaptive exit counts:
   ```python
   exits_signal: int       # kept for backward compat (old config)
   exits_hard_sl: int
   exits_breakeven: int
   exits_tier1: int
   exits_tier2: int
   exits_tier3: int
   exits_velocity: int
   exits_trail: int        # legacy (old config)
   exits_max_hold: int
   exits_session_end: int
   ```

2. Update `run_backtest()` signature to accept `PositionManagerConfig | AdaptiveStopConfig | None`.

3. Update `_compute_stats()` to count all new exit reasons.

4. Update zero-trade `BacktestResult` in `_compute_stats` to include new fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_adaptive_stop.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: all existing tests PASS (backward compatible)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/engine.py tests/test_adaptive_stop.py
git commit -m "feat: update BacktestResult for adaptive stop exit reasons"
```

---

### Task 4: Random config sampler with constraint enforcement

**Files:**
- Create: `scripts/sweep_adaptive_stop.py` (initial structure with sampler only)
- Test: `tests/test_adaptive_stop.py`

- [ ] **Step 1: Write failing tests for config sampler**

Append to `tests/test_adaptive_stop.py`:

```python
# Import will be at top of file after creation
# from scripts.sweep_adaptive_stop import sample_valid_config, PARAM_RANGES


class TestConfigSampler:
    def test_sample_returns_valid_config(self):
        """Every sampled config passes validation."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from sweep_adaptive_stop import sample_valid_config

        rng = np.random.default_rng(42)
        for _ in range(200):
            cfg = sample_valid_config(rng)
            cfg.validate()  # must not raise

    def test_sample_diversity(self):
        """Samples produce diverse configs, not all identical."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from sweep_adaptive_stop import sample_valid_config

        rng = np.random.default_rng(42)
        hard_sls = set()
        for _ in range(100):
            cfg = sample_valid_config(rng)
            hard_sls.add(cfg.hard_sl_ticks)
        assert len(hard_sls) >= 3  # at least 3 different SL values

    def test_disabled_mechanics_possible(self):
        """Some samples should have mechanics disabled (0 values)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from sweep_adaptive_stop import sample_valid_config

        rng = np.random.default_rng(42)
        has_no_breakeven = False
        has_no_trail = False
        has_no_velocity = False
        for _ in range(200):
            cfg = sample_valid_config(rng)
            if cfg.breakeven_trigger_ticks == 0:
                has_no_breakeven = True
            if cfg.tier1_activation_ticks == 0:
                has_no_trail = True
            if cfg.velocity_lookback_bars == 0:
                has_no_velocity = True
        assert has_no_breakeven and has_no_trail and has_no_velocity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_adaptive_stop.py::TestConfigSampler -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the sampler in sweep script**

Create `scripts/sweep_adaptive_stop.py` with:

```python
"""Adaptive stop parameter sweep with walk-forward validation.

Random search over AdaptiveStopConfig parameter space per regime pair.
Evaluates via walk-forward: optimize on expanding train, test on next month.

Usage:
    python scripts/sweep_adaptive_stop.py
    python scripts/sweep_adaptive_stop.py --year 2025 --n-samples 500
    python scripts/sweep_adaptive_stop.py --models pair_3_0 pair_1_0
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.position_manager import AdaptiveStopConfig, MES_TICK_VALUE
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import (
    _compute_features, _filter_rth, _resample_to_1sec, _z_score_normalize,
)
from src.models.entry_model import EntryModel
from src.regime.macro_features_v2 import macro_features_from_1s_bars
from src.regime.micro_features_v2 import micro_features_from_1s_bars

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)

COMMISSION_RT = 0.59
MAX_HOLD = 300
MIN_TRAIN_TRADES = 50
MIN_OOS_TRADES = 30

# Entry thresholds swept separately
ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Parameter ranges for random sampling
PARAM_RANGES = {
    "hard_sl_ticks": [4, 6, 8, 10, 12],
    "breakeven_trigger_ticks": [0, 2, 3, 4, 6],
    "breakeven_lock_ticks": [0.5, 1.0, 1.5, 2.0],
    "tier1_activation_ticks": [0, 2, 3, 4, 6],
    "tier1_trail_distance": [1.0, 1.5, 2.0, 3.0, 4.0],
    "tier2_activation_ticks": [0, 5, 6, 8, 10],
    "tier2_trail_distance": [0.5, 1.0, 1.5, 2.0],
    "tier3_activation_ticks": [0, 8, 10, 12, 15],
    "tier3_trail_distance": [0.25, 0.5, 1.0],
    "velocity_lookback_bars": [0, 2, 3, 5, 10],
    "velocity_threshold_ticks": [3, 4, 6, 8],
    "velocity_trail_distance": [0.25, 0.5, 1.0],
}


def sample_valid_config(rng: np.random.Generator) -> AdaptiveStopConfig:
    """Sample a random valid AdaptiveStopConfig.

    Enforces all constraints from the spec:
    - Tier activations strictly increasing (when non-zero)
    - Tier distances strictly decreasing (when active)
    - Disabled tier disables higher tiers
    - Sub-params only sampled when parent mechanic active
    """
    pick = lambda key: rng.choice(PARAM_RANGES[key])

    hard_sl = float(pick("hard_sl_ticks"))

    # Breakeven
    be_trigger = float(pick("breakeven_trigger_ticks"))
    be_lock = float(pick("breakeven_lock_ticks")) if be_trigger > 0 else 0.0

    # Tier 1
    t1_act = float(pick("tier1_activation_ticks"))
    t1_dist = float(pick("tier1_trail_distance")) if t1_act > 0 else 2.0

    # Tier 2 (only if tier 1 active)
    if t1_act > 0:
        t2_act_choices = [v for v in PARAM_RANGES["tier2_activation_ticks"]
                          if v == 0 or v > t1_act]
        t2_act = float(rng.choice(t2_act_choices)) if t2_act_choices else 0.0
    else:
        t2_act = 0.0

    if t2_act > 0:
        t2_dist_choices = [v for v in PARAM_RANGES["tier2_trail_distance"]
                           if v < t1_dist]
        t2_dist = float(rng.choice(t2_dist_choices)) if t2_dist_choices else t1_dist / 2
    else:
        t2_dist = 1.0

    # Tier 3 (only if tier 2 active)
    if t2_act > 0:
        t3_act_choices = [v for v in PARAM_RANGES["tier3_activation_ticks"]
                          if v == 0 or v > t2_act]
        t3_act = float(rng.choice(t3_act_choices)) if t3_act_choices else 0.0
    else:
        t3_act = 0.0

    if t3_act > 0:
        t3_dist_choices = [v for v in PARAM_RANGES["tier3_trail_distance"]
                           if v < t2_dist]
        t3_dist = float(rng.choice(t3_dist_choices)) if t3_dist_choices else t2_dist / 2
    else:
        t3_dist = 0.5

    # Velocity
    vel_lookback = int(pick("velocity_lookback_bars"))
    vel_threshold = float(pick("velocity_threshold_ticks")) if vel_lookback > 0 else 0.0
    vel_trail = float(pick("velocity_trail_distance")) if vel_lookback > 0 else 0.5

    return AdaptiveStopConfig(
        hard_sl_ticks=hard_sl,
        breakeven_trigger_ticks=be_trigger,
        breakeven_lock_ticks=be_lock,
        tier1_activation_ticks=t1_act,
        tier1_trail_distance=t1_dist,
        tier2_activation_ticks=t2_act,
        tier2_trail_distance=t2_dist,
        tier3_activation_ticks=t3_act,
        tier3_trail_distance=t3_dist,
        velocity_lookback_bars=vel_lookback,
        velocity_threshold_ticks=vel_threshold,
        velocity_trail_distance=vel_trail,
        max_hold_bars=MAX_HOLD,
        commission_rt_dollars=COMMISSION_RT,
    )
```

(This is just the sampler portion. The full sweep logic comes in Task 5.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_adaptive_stop.py::TestConfigSampler -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_adaptive_stop.py tests/test_adaptive_stop.py
git commit -m "feat: random config sampler with constraint enforcement"
```

---

### Task 5: Walk-forward sweep engine

**Files:**
- Modify: `scripts/sweep_adaptive_stop.py`

This task adds the main sweep loop. It reuses infrastructure from the existing `sweep_walkforward.py` (data loading, regime labels, month boundaries, session breaks) but replaces the grid search with random sampling and adds the new scoring function.

- [ ] **Step 1: Add helper functions**

Add to `scripts/sweep_adaptive_stop.py` — reuse `_load_bars`, `_compute_regime_labels`, `_get_month_boundaries`, `_adjust_session_breaks` from `scripts/sweep_walkforward.py` by importing from a shared location or copying. Since these are script-level helpers, copy them (they're stable and changing the old script is risky).

Key additions:

```python
def _score(result: BacktestResult) -> float:
    """Score a backtest result: net_pnl / (1 + max_drawdown_ticks)."""
    if result.n_trades < MIN_TRAIN_TRADES:
        return float("-inf")
    return result.net_pnl_dollars / (1.0 + result.max_drawdown_ticks)


def _sweep_fold(
    p_up: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[AdaptiveStopConfig | None, float, list[dict]]:
    """Random search on a train fold. Returns (best_config, best_score, all_results)."""
    best_config = None
    best_score = float("-inf")
    all_results = []

    for entry_t in ENTRY_THRESHOLDS:
        for _ in range(n_samples // len(ENTRY_THRESHOLDS)):
            cfg = sample_valid_config(rng)
            cfg.long_entry = entry_t
            cfg.short_entry = 1.0 - entry_t

            result = run_backtest(p_up, mid, bid, ask, session_breaks, cfg)
            score = _score(result)

            row = asdict(cfg)
            row["n_trades"] = result.n_trades
            row["win_rate"] = result.win_rate
            row["net_pnl_dollars"] = result.net_pnl_dollars
            row["max_drawdown_ticks"] = result.max_drawdown_ticks
            row["profit_factor"] = result.profit_factor
            row["score"] = score
            all_results.append(row)

            if score > best_score:
                best_score = score
                best_config = cfg

    return best_config, best_score, all_results
```

- [ ] **Step 2: Add the main CLI with walk-forward loop**

Add to `scripts/sweep_adaptive_stop.py`:

```python
@click.command()
@click.option("--year", default=2025, type=int)
@click.option("--models", multiple=True, default=None)
@click.option("--window-size", default=30, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--batch-size", default=4096, type=int)
@click.option("--n-samples", default=1000, type=int, help="Random samples per fold")
@click.option("--seed", default=42, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(year, models, window_size, model_dir, batch_size, n_samples, seed, verbose):
    """Adaptive stop parameter sweep with walk-forward validation."""
    # (mirrors sweep_walkforward.py structure)
    # 1. Load data, compute features, regime labels
    # 2. For each tradeable model:
    #    a. Run rolling inference, gate by regime
    #    b. Walk-forward folds:
    #       - _sweep_fold on train -> best config
    #       - Evaluate best on test (OOS)
    # 3. Collect all results, save outputs
```

The full main function follows the same structure as `sweep_walkforward.py` lines 300-594, but calls `_sweep_fold()` instead of `_run_sweep_on_slice()`, and uses `AdaptiveStopConfig` instead of `PositionManagerConfig`.

- [ ] **Step 3: Add output saving**

At the end of `main()`, save:
- `results/adaptive_stop_sweep/YYYY-MM-DD_HHMMSS/sweep_results.json`
- `results/adaptive_stop_sweep/YYYY-MM-DD_HHMMSS/best_configs.json`
- `results/adaptive_stop_sweep/YYYY-MM-DD_HHMMSS/sweep_log.csv`

```python
    # Save results
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "adaptive_stop_sweep" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(out_dir / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2, default=str)

    # Flatten sweep log to CSV
    if all_sweep_rows:
        pd.DataFrame(all_sweep_rows).to_csv(out_dir / "sweep_log.csv", index=False)

    click.echo(f"\n  Results saved to {out_dir}")
```

- [ ] **Step 4: Smoke test with dry run**

Run: `python scripts/sweep_adaptive_stop.py --year 2025 --n-samples 10 --models pair_3_0 -v`
Expected: Runs without error, produces output files in `results/adaptive_stop_sweep/`

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_adaptive_stop.py
git commit -m "feat: walk-forward adaptive stop sweep with random search"
```

---

### Task 6: Robustness checks (sensitivity, surface flatness, Monte Carlo)

**Files:**
- Modify: `scripts/sweep_adaptive_stop.py`

- [ ] **Step 1: Add sensitivity analysis function**

```python
def _sensitivity_analysis(
    base_config: AdaptiveStopConfig,
    p_up: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
) -> dict:
    """Perturb each param +/- 1 step, report score change."""
    base_result = run_backtest(p_up, mid, bid, ask, session_breaks, base_config)
    base_score = _score(base_result)
    perturbations = {}

    for param, values in PARAM_RANGES.items():
        current = getattr(base_config, param)
        if current not in values:
            continue
        idx = values.index(current)
        scores = {}

        for direction, offset in [("down", -1), ("up", 1)]:
            new_idx = idx + offset
            if new_idx < 0 or new_idx >= len(values):
                continue
            cfg = AdaptiveStopConfig(**{
                **asdict(base_config),
                param: values[new_idx],
            })
            try:
                cfg.validate()
            except ValueError:
                continue
            r = run_backtest(p_up, mid, bid, ask, session_breaks, cfg)
            scores[direction] = _score(r)

        perturbations[param] = {
            "base_score": base_score,
            **scores,
        }

    return perturbations
```

- [ ] **Step 2: Add Monte Carlo permutation test**

```python
def _monte_carlo_test(
    trade_pnls: list[float],
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Shuffle trade P&Ls, check if actual net beats 95th percentile."""
    rng = rng or np.random.default_rng(42)
    pnls = np.array(trade_pnls)
    actual_net = pnls.sum()

    shuffled_nets = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled_nets[i] = (pnls * signs).sum()

    p95 = np.percentile(shuffled_nets, 95)
    p_value = (shuffled_nets >= actual_net).mean()

    return {
        "actual_net_pnl": float(actual_net),
        "p95_threshold": float(p95),
        "p_value": float(p_value),
        "significant": bool(actual_net > p95),
    }
```

- [ ] **Step 3: Add surface flatness calculation**

```python
def _surface_flatness(all_scores: list[float], best_score: float) -> dict:
    """Count configs within 80% of best score."""
    # Handle negative scores: 80% of a negative number is MORE negative,
    # so flip the multiplier to get a LESS negative threshold.
    if best_score > 0:
        threshold = best_score * 0.8
    else:
        threshold = best_score * 1.2  # e.g., -100 * 1.2 = -120 (more lenient)
    n_good = sum(1 for s in all_scores if s >= threshold)
    return {
        "best_score": best_score,
        "threshold_80pct": threshold,
        "n_good_configs": n_good,
        "n_total_configs": len(all_scores),
        "pct_good": n_good / len(all_scores) * 100 if all_scores else 0,
    }
```

- [ ] **Step 4: Wire robustness checks into main, save to robustness_report.json**

After walk-forward completes for each model, run:
1. Sensitivity analysis on OOS data with best config
2. Surface flatness from the sweep log scores
3. Monte Carlo test on OOS trade P&Ls

Save to `out_dir / "robustness_report.json"`.

- [ ] **Step 5: Smoke test**

Run: `python scripts/sweep_adaptive_stop.py --year 2025 --n-samples 10 --models pair_3_0 -v`
Expected: `robustness_report.json` created with sensitivity, flatness, and Monte Carlo sections

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_adaptive_stop.py
git commit -m "feat: robustness checks — sensitivity, surface flatness, Monte Carlo"
```

---

### Task 7: End-to-end validation

**Files:**
- No new files. Validation only.

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Run sweep on real data with small sample size**

Run: `python scripts/sweep_adaptive_stop.py --year 2025 --n-samples 50 --models pair_3_0 -v`
Expected: Completes, prints OOS results, saves all 4 output files

- [ ] **Step 3: Verify output files**

Check `results/adaptive_stop_sweep/<timestamp>/`:
- `sweep_results.json` — has per-fold data with OOS metrics
- `best_configs.json` — has config for pair_3_0
- `sweep_log.csv` — has rows with all config params + metrics
- `robustness_report.json` — has sensitivity, flatness, Monte Carlo

- [ ] **Step 4: Spot-check a winning config**

Load `best_configs.json`, create an `AdaptiveStopConfig` from it, call `.validate()`. Verify it doesn't raise.

- [ ] **Step 5: Commit any fixes, tag as complete**

```bash
git add -A
git commit -m "chore: end-to-end validation of adaptive stop sweep"
```
