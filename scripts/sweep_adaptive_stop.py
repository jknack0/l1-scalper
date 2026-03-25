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

ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

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

    Enforces all constraints:
    - Tier activations strictly increasing (when non-zero)
    - Tier distances strictly decreasing (when active)
    - Disabled tier disables higher tiers
    - Sub-params only sampled when parent mechanic active
    """
    pick = lambda key: rng.choice(PARAM_RANGES[key])

    hard_sl = float(pick("hard_sl_ticks"))

    be_trigger = float(pick("breakeven_trigger_ticks"))
    be_lock = float(pick("breakeven_lock_ticks")) if be_trigger > 0 else 0.0

    t1_act = float(pick("tier1_activation_ticks"))
    t1_dist = float(pick("tier1_trail_distance")) if t1_act > 0 else 2.0

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
