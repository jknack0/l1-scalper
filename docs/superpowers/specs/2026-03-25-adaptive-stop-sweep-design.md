# Adaptive Stop Sweep Design

## Problem

The current trailing stop has fixed activation (6 ticks) and distance (0.5 ticks). When a trade rips fast (e.g., 10 ticks in 5 seconds), the system doesn't adapt — it can't tighten the trail based on momentum or lock in breakeven after covering commission. Different regime pairs likely have different optimal exit mechanics, but we have no way to discover this without sweeping.

## Solution

A unified "adaptive stop" that composes four exit mechanics — hard SL, breakeven lock, 3-tier trailing stop, and velocity-based ratchet. At each bar, the stop price is the tightest (most protective) of all active mechanics. All parameters are sweepable per regime pair using random search within a walk-forward framework.

## Adaptive Stop Config

```python
@dataclass
class AdaptiveStopConfig:
    # --- Hard stop ---
    hard_sl_ticks: float = 8.0

    # --- Breakeven lock ---
    breakeven_trigger_ticks: float = 0.0   # profit to activate breakeven (0 = disabled)
    breakeven_lock_ticks: float = 0.0      # stop moves to entry + this value

    # --- 3-Tier trailing stop ---
    tier1_activation_ticks: float = 0.0    # profit to enter tier 1 (0 = no trail)
    tier1_trail_distance: float = 2.0
    tier2_activation_ticks: float = 0.0    # profit to enter tier 2 (0 = skip)
    tier2_trail_distance: float = 1.0
    tier3_activation_ticks: float = 0.0    # profit to enter tier 3 (0 = skip)
    tier3_trail_distance: float = 0.5

    # --- Velocity ratchet ---
    velocity_lookback_bars: int = 0        # seconds to look back (0 = disabled)
    velocity_threshold_ticks: float = 0.0  # tick move that triggers ratchet
    velocity_trail_distance: float = 0.5   # trail distance when velocity triggered
```

### Constraints

- tier1_activation < tier2_activation < tier3_activation (when non-zero)
- tier1_trail_distance > tier2_trail_distance > tier3_trail_distance
- breakeven_lock only relevant when breakeven_trigger > 0
- velocity params only relevant when velocity_lookback > 0
- If a tier is disabled (0), higher tiers must also be disabled

## Stop Price Computation

Each bar, compute a single stop level (in ticks relative to entry) by taking the max of all active mechanics:

```
Every bar:
  current_pnl = (mid - entry) * direction  # in ticks
  mfe = max(mfe, current_pnl)              # track best profit

  # 1. Hard SL (always)
  stop_level = -hard_sl_ticks

  # 2. Breakeven lock
  if mfe >= breakeven_trigger_ticks and breakeven_trigger_ticks > 0:
      stop_level = max(stop_level, breakeven_lock_ticks)

  # 3. Tiered trail (highest active tier wins)
  if mfe >= tier3_activation_ticks and tier3_activation_ticks > 0:
      stop_level = max(stop_level, mfe - tier3_trail_distance)
  elif mfe >= tier2_activation_ticks and tier2_activation_ticks > 0:
      stop_level = max(stop_level, mfe - tier2_trail_distance)
  elif mfe >= tier1_activation_ticks and tier1_activation_ticks > 0:
      stop_level = max(stop_level, mfe - tier1_trail_distance)

  # 4. Velocity ratchet
  if velocity_lookback_bars > 0:
      recent_move = current_pnl - pnl_N_bars_ago
      if recent_move >= velocity_threshold_ticks:
          stop_level = max(stop_level, mfe - velocity_trail_distance)

  # Exit check
  if current_pnl <= stop_level:
      exit("adaptive_stop")
```

The binding mechanic (whichever produced the highest stop_level) is recorded as the exit sub-reason for analysis.

## Position Manager Changes

### Modified

- Exit cascade becomes: adaptive stop -> max hold -> session end
- Signal-based exits are removed (adaptive stop handles all profit-taking)
- A ring buffer (numpy array, size = max velocity_lookback_bars) tracks recent P&L for velocity calculation
- Exit reason captures binding mechanic: `hard_sl`, `breakeven`, `tier1`, `tier2`, `tier3`, `velocity`, `max_hold`, `session_end`

### Unchanged

- Entry logic (P(up) thresholds, regime gating)
- Fill model (entry at ask/bid, exit at bid/ask)
- FLAT -> LONG/SHORT -> FLAT state machine
- Trade record structure (add exit_sub_reason field)

### Backward Compatibility

Old configs map to: tier1_activation = trail_activation_ticks, tier1_trail_distance = trail_distance_ticks, everything else disabled.

## Sweep Design

### Method: Random Search

1,000 random valid configs per regime pair, per walk-forward fold. Random search over grid search because the parameter space is high-dimensional (13 params) and research shows random search finds good configs faster.

### Parameter Ranges

| Parameter | Values |
|---|---|
| hard_sl_ticks | 4, 6, 8, 10, 12 |
| breakeven_trigger_ticks | 0, 2, 3, 4, 6 |
| breakeven_lock_ticks | 0, 0.5, 1.0, 1.5, 2.0 |
| tier1_activation_ticks | 0, 2, 3, 4, 6 |
| tier1_trail_distance | 1.0, 1.5, 2.0, 3.0, 4.0 |
| tier2_activation_ticks | 0, 5, 6, 8, 10 |
| tier2_trail_distance | 0.5, 1.0, 1.5, 2.0 |
| tier3_activation_ticks | 0, 8, 10, 12, 15 |
| tier3_trail_distance | 0.25, 0.5, 1.0 |
| velocity_lookback_bars | 0, 2, 3, 5, 10 |
| velocity_threshold_ticks | 3, 4, 6, 8 |
| velocity_trail_distance | 0.25, 0.5, 1.0 |

### Constraint Enforcement During Sampling

- Tier activations must be strictly increasing (when non-zero)
- Tier distances must be strictly decreasing
- Disabled tiers disable all higher tiers
- Breakeven/velocity sub-params only sampled when their triggers are active
- Invalid samples are rejected and re-sampled

### Walk-Forward Process

1. For each tradeable regime pair, load its model and run regime-gated rolling inference
2. Expanding train window, 1-month test window
3. On each train fold: sample 1,000 random valid configs, backtest each, rank by net P&L (after commission at $0.70 RT)
4. Best train config evaluated on test fold (OOS)
5. Aggregate OOS results across folds per regime pair

### Output

Saved to `results/adaptive_stop_sweep/YYYY-MM-DD_HHMMSS/`:

- **`sweep_results.json`** — full results: per regime pair, per fold, best config, OOS metrics, exit reason breakdown
- **`best_configs.json`** — winning config per regime pair (loadable by live bot)
- **`sweep_log.csv`** — every sampled config + train/test performance (flat CSV for pandas/Excel analysis)

### Runtime Estimate

~9 tradeable regime pairs x 1,000 samples x 6-8 folds = 50K-70K backtests. Each backtest is vectorized and fast. Estimated 30-60 minutes total.

## Live Bot Integration Path

After sweep identifies winning configs:

1. `best_configs.json` maps each regime pair to its AdaptiveStopConfig
2. `RegimeModelConfig` in the live bot gets the same adaptive stop fields
3. Hard SL stays as a server-side Tradovate bracket order (crash protection)
4. Adaptive stop tightening managed client-side via cancel/replace orders
5. No live bot changes in this phase — sweep first, port winners later

## Sensitivity / Robustness Check

For each regime pair's winning config, perturb each parameter by +/- 1 step and re-run the backtest. If performance degrades sharply, the config is likely overfit. Only trust configs that degrade gracefully. Results included in sweep_results.json.
