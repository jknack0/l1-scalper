"""Sweep bracket + trailing stop configurations.

Same z-score entry as bracket sweep, but adds a trailing stop that
activates after price moves N ticks in your favor. Once active, the
stop trails at a fixed distance behind the best price.

Usage:
    python scripts/sweep_trailing.py --year 2026 --model-name pair_3_0 --regime 3_0
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.position_manager import MES_TICK, MES_TICK_VALUE, Side
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import _compute_features, _filter_rth, _resample_to_1sec, _z_score_normalize
from src.models.entry_model import EntryModel
from src.regime.macro_features_v2 import DEFAULT_MACRO_WINDOW, macro_features_from_1s_bars
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_features_v2 import DEFAULT_MICRO_WINDOW, micro_features_from_1s_bars
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)


def _rolling_zscore(p_up: np.ndarray, lookback: int = 300) -> np.ndarray:
    n = len(p_up)
    z = np.full(n, np.nan, dtype=np.float32)
    valid = np.isfinite(p_up)
    p_clean = np.where(valid, p_up, 0.0)
    cs = np.cumsum(p_clean)
    cs2 = np.cumsum(p_clean ** 2)
    cv = np.cumsum(valid.astype(np.float64))
    for i in range(lookback, n):
        if not valid[i]:
            continue
        j = i - lookback
        count = cv[i] - cv[j]
        if count < 30:
            continue
        s = cs[i] - cs[j]
        s2 = cs2[i] - cs2[j]
        mean = s / count
        var = s2 / count - mean ** 2
        if var < 1e-10:
            continue
        z[i] = (p_up[i] - mean) / np.sqrt(var)
    return z


def _run_trailing_backtest(
    z_scores, mid, bid, ask, session_breaks,
    entry_z, sl_ticks, trail_activation, trail_distance,
    max_hold, min_hold, commission_rt, regime_mask=None,
):
    """Bracket with trailing stop.

    - Entry: z >= entry_z (long) or z <= -entry_z (short)
    - Hard SL: fixed stop loss from entry
    - Trailing: once price moves trail_activation ticks in your favor,
      a trailing stop activates at trail_distance ticks behind best price.
      No fixed TP — the trail lets winners run.
    - Max hold: time-based exit
    """
    n = len(z_scores)
    session_set = set(session_breaks) if session_breaks is not None else set()

    trades = []
    in_trade = False
    side = Side.FLAT
    entry_bar = 0
    entry_fill = 0.0
    best_price = 0.0  # best favorable price since entry
    trail_active = False
    cooldown_until = 0

    for i in range(n):
        if i in session_set:
            if in_trade:
                pnl = _pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": i - entry_bar, "exit": "session",
                               "best_runup": _runup(side, entry_fill, best_price)})
                in_trade = False
                side = Side.FLAT
            cooldown_until = i + min_hold

        if in_trade:
            hold = i - entry_bar

            # Track best price
            if side == Side.LONG:
                best_price = max(best_price, mid[i])
                current_pnl = (mid[i] - entry_fill) / MES_TICK
                runup = (best_price - entry_fill) / MES_TICK
            else:
                best_price = min(best_price, mid[i])
                current_pnl = (entry_fill - mid[i]) / MES_TICK
                runup = (entry_fill - best_price) / MES_TICK

            # Hard SL (always active)
            if current_pnl <= -sl_ticks:
                pnl = _pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": hold, "exit": "sl",
                               "best_runup": runup})
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue

            # Activate trailing stop
            if not trail_active and runup >= trail_activation:
                trail_active = True

            # Check trailing stop
            if trail_active:
                drawback = runup - current_pnl  # how far price has pulled back from best
                if drawback >= trail_distance:
                    pnl = _pnl(side, entry_fill, bid[i], ask[i])
                    trades.append({"pnl": pnl, "hold": hold, "exit": "trail",
                                   "best_runup": runup})
                    in_trade = False
                    side = Side.FLAT
                    trail_active = False
                    cooldown_until = i + min_hold
                    continue

            # Max hold
            if hold >= max_hold:
                pnl = _pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": hold, "exit": "max_hold",
                               "best_runup": runup})
                in_trade = False
                side = Side.FLAT
                trail_active = False
                cooldown_until = i + min_hold
                continue

        else:
            if i < cooldown_until:
                continue
            if regime_mask is not None and not regime_mask[i]:
                continue
            z = z_scores[i]
            if np.isnan(z):
                continue
            if z >= entry_z:
                in_trade = True
                side = Side.LONG
                entry_bar = i
                entry_fill = ask[i]
                best_price = mid[i]
                trail_active = False
            elif z <= -entry_z:
                in_trade = True
                side = Side.SHORT
                entry_bar = i
                entry_fill = bid[i]
                best_price = mid[i]
                trail_active = False

    if in_trade:
        pnl = _pnl(side, entry_fill, bid[n-1], ask[n-1])
        runup = _runup(side, entry_fill, best_price)
        trades.append({"pnl": pnl, "hold": n - 1 - entry_bar, "exit": "session",
                       "best_runup": runup})

    return _summarize(trades, commission_rt)


def _pnl(side, entry_fill, bid, ask):
    if side == Side.LONG:
        return (bid - entry_fill) / MES_TICK
    else:
        return (entry_fill - ask) / MES_TICK


def _runup(side, entry_fill, best_price):
    if side == Side.LONG:
        return (best_price - entry_fill) / MES_TICK
    else:
        return (entry_fill - best_price) / MES_TICK


def _summarize(trades, commission_rt):
    if not trades:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    holds = np.array([t["hold"] for t in trades])
    runups = np.array([t["best_runup"] for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    gp = winners.sum() if len(winners) > 0 else 0.0
    gl = abs(losers.sum()) if len(losers) > 0 else 0.0
    cum = np.cumsum(pnls)
    max_dd = (np.maximum.accumulate(cum) - cum).max()
    n = len(trades)
    n_long = sum(1 for t in trades if t.get("side") == Side.LONG)
    exits = [t["exit"] for t in trades]

    return {
        "n": n, "n_long": sum(1 for t in trades if "side" not in t or True),
        "wr": len(winners) / n, "avg_pnl": float(pnls.mean()),
        "avg_win": float(winners.mean()) if len(winners) > 0 else 0.0,
        "avg_loss": float(losers.mean()) if len(losers) > 0 else 0.0,
        "pf": gp / gl if gl > 0 else float("inf"),
        "net": float(pnls.sum()) * MES_TICK_VALUE - n * commission_rt,
        "max_dd": float(max_dd), "avg_hold": float(holds.mean()),
        "avg_runup": float(runups.mean()),
        "sl": exits.count("sl"), "trail": exits.count("trail"),
        "max_hold": exits.count("max_hold"), "session": exits.count("session"),
    }


@click.command()
@click.option("--year", default=2026, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--model-name", default="pair_3_0", type=str)
@click.option("--regime", default=None, type=str, help="e.g., '3_0'")
@click.option("--macro-hmm-path", default="models/macro_hmm_v2.pkl")
@click.option("--micro-hmm-path", default="models/micro_hmm_v2.pkl")
@click.option("--window-size", default=30, type=int)
@click.option("--lookback", default=300, type=int)
@click.option("--max-hold", default=300, type=int)
@click.option("--min-hold", default=30, type=int)
@click.option("--commission", default=0.59, type=float)
@click.option("--batch-size", default=4096, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    year, model_dir, model_name, regime, macro_hmm_path, micro_hmm_path,
    window_size, lookback, max_hold, min_hold, commission, batch_size, verbose,
):
    """Sweep trailing stop configs for a regime-gated model."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}_h5"
    else:
        model_dir_path = Path(model_dir)

    model_path = model_dir_path / f"{model_name}.pt"

    click.echo(f"Model: {model_path}")
    click.echo(f"Year: {year}, Regime: {regime}")
    click.echo()

    # ── Load data ─────────────────────────────────────────────────
    click.echo("[1/3] LOADING DATA...")
    l1_path = L1_DIR / f"year={year}" / "data.parquet"
    pf = pq.ParquetFile(l1_path)
    df = pf.read().to_pandas()
    bars = _resample_to_1sec(df)
    del df
    gc.collect()

    bars = _filter_rth(bars)
    bar_seconds = bars.index.values
    gaps = np.diff(bar_seconds)
    session_breaks = np.where(gaps > 60)[0] + 1

    raw_features = _compute_features(bars)
    features = _z_score_normalize(raw_features)
    del raw_features
    gc.collect()

    mid = bars["mid"].values.astype(np.float32)
    bid = bars["bid"].values.astype(np.float32)
    ask = bars["ask"].values.astype(np.float32)
    n_bars = len(mid)

    # Regime mask
    regime_mask = None
    if regime and macro_hmm_path and micro_hmm_path:
        target_macro, target_micro = int(regime.split("_")[0]), int(regime.split("_")[1])
        bars_df_macro = pd.DataFrame({
            "close": bars["mid"].values, "volume": bars["total_vol"].values,
            "timestamp": bars.index.values,
        })
        macro_feats, _ = macro_features_from_1s_bars(bars_df_macro, window=DEFAULT_MACRO_WINDOW)
        bars_df_micro = pd.DataFrame({
            "close": bars["mid"].values, "volume": bars["total_vol"].values,
            "spread_ticks": bars["spread_ticks"].values,
            "ofi": (np.diff(bars["bid_sz"].values, prepend=bars["bid_sz"].values[0])
                    - np.diff(bars["ask_sz"].values, prepend=bars["ask_sz"].values[0])),
            "timestamp": bars.index.values,
        })
        micro_feats, _ = micro_features_from_1s_bars(bars_df_micro, window=DEFAULT_MICRO_WINDOW)

        macro_hmm = MacroRegimeHMMv2()
        macro_hmm.load(macro_hmm_path)
        micro_hmm = MicroRegimeHMMv2()
        micro_hmm.load(micro_hmm_path)

        macro_labels = macro_hmm.predict_proba_forward(macro_hmm.normalize(macro_feats)).argmax(axis=1)
        micro_labels = micro_hmm.predict_proba_forward(micro_hmm.normalize(micro_feats)).argmax(axis=1)

        macro_per_bar = np.repeat(macro_labels, DEFAULT_MACRO_WINDOW)[:n_bars]
        micro_per_bar = np.repeat(micro_labels, DEFAULT_MICRO_WINDOW)[:n_bars]
        if len(macro_per_bar) < n_bars:
            macro_per_bar = np.pad(macro_per_bar, (0, n_bars - len(macro_per_bar)), constant_values=-1)
        if len(micro_per_bar) < n_bars:
            micro_per_bar = np.pad(micro_per_bar, (0, n_bars - len(micro_per_bar)), constant_values=-1)

        regime_mask = (macro_per_bar == target_macro) & (micro_per_bar == target_micro)
        click.echo(f"  Regime ({target_macro},{target_micro}): {regime_mask.sum():,} bars ({regime_mask.mean()*100:.1f}%)")

    del bars
    gc.collect()

    # ── Inference ─────────────────────────────────────────────────
    click.echo("\n[2/3] ROLLING INFERENCE...")
    model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    p_up = rolling_inference(model, features, window_size=window_size,
                             batch_size=batch_size, device=device)
    z = _rolling_zscore(p_up, lookback=lookback)

    del model, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Sweep ─────────────────────────────────────────────────────
    click.echo(f"\n[3/3] SWEEPING TRAILING STOP CONFIGS...")

    entry_zs = [2.0, 2.5, 3.0]
    sl_ticks_list = [6, 8, 9, 12]
    trail_activations = [4, 6, 8, 9, 12]
    trail_distances = [2, 3, 4, 6]

    click.echo(f"\n  {'EZ':>3} | {'SL':>3} | {'TrAct':>5} | {'TrDist':>6} | {'Trades':>6} | "
               f"{'WR':>5} | {'AvgPnL':>7} | {'AvgWin':>6} | {'PF':>5} | {'Net$':>9} | "
               f"{'MaxDD':>5} | {'Hold':>5} | {'Runup':>5} | {'SL%':>4} | {'Tr%':>4} | {'MH%':>4}")
    click.echo(f"  {'-'*3}-+-{'-'*3}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-"
               f"{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}-+-"
               f"{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}")

    best_net = float("-inf")
    best_params = None

    for entry_z in entry_zs:
        for sl in sl_ticks_list:
            for t_act in trail_activations:
                for t_dist in trail_distances:
                    if t_dist >= t_act:
                        continue  # trail distance must be less than activation

                    r = _run_trailing_backtest(
                        z, mid, bid, ask, session_breaks,
                        entry_z=entry_z, sl_ticks=float(sl),
                        trail_activation=float(t_act), trail_distance=float(t_dist),
                        max_hold=max_hold, min_hold=min_hold,
                        commission_rt=commission, regime_mask=regime_mask,
                    )

                    if r is None or r["n"] < 10:
                        continue

                    n = r["n"]
                    sl_pct = r["sl"] / n * 100
                    tr_pct = r["trail"] / n * 100
                    mh_pct = r["max_hold"] / n * 100

                    # Only print interesting configs
                    if r["net"] > -100:
                        click.echo(
                            f"  {entry_z:>3.1f} | {sl:>3} | {t_act:>5} | {t_dist:>6} | "
                            f"{n:>6} | {r['wr']:>5.1%} | {r['avg_pnl']:>+7.2f} | "
                            f"{r['avg_win']:>6.2f} | {r['pf']:>5.2f} | {r['net']:>+9.2f} | "
                            f"{r['max_dd']:>5.0f} | {r['avg_hold']:>5.0f}s | "
                            f"{r['avg_runup']:>5.1f} | {sl_pct:>4.0f} | {tr_pct:>4.0f} | {mh_pct:>4.0f}"
                        )

                    if r["net"] > best_net:
                        best_net = r["net"]
                        best_params = {**r, "entry_z": entry_z, "sl": sl,
                                       "trail_act": t_act, "trail_dist": t_dist}

    if best_params:
        b = best_params
        click.echo(f"\n  BEST: z={b['entry_z']}, SL={b['sl']}, trail_act={b['trail_act']}, trail_dist={b['trail_dist']}")
        click.echo(f"    {b['n']} trades, WR={b['wr']:.1%}, AvgWin={b['avg_win']:.2f}t, "
                   f"PF={b['pf']:.2f}, Net=${b['net']:+,.2f}, MaxDD={b['max_dd']:.0f}t, "
                   f"Hold={b['avg_hold']:.0f}s, AvgRunup={b['avg_runup']:.1f}t")


if __name__ == "__main__":
    main()
