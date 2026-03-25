"""Sweep all regime-specific models on their respective regimes.

For each pair_X_Y model, runs the bracket sweep gated to regime (X, Y).
Collects the best config per model and prints a combined summary.

Usage:
    python scripts/sweep_all_regimes.py --year 2026
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

from src.backtest.position_manager import MES_TICK, MES_TICK_VALUE, Side, Trade
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


def _bracket_pnl(side: Side, entry_fill: float, bid: float, ask: float) -> float:
    if side == Side.LONG:
        return (bid - entry_fill) / MES_TICK
    else:
        return (entry_fill - ask) / MES_TICK


def _run_bracket_backtest(
    z_scores, mid, bid, ask, session_breaks,
    entry_z, tp_ticks, sl_ticks, max_hold, min_hold, commission_rt,
    regime_mask=None,
):
    n = len(z_scores)
    session_set = set(session_breaks) if session_breaks is not None else set()
    trades = []
    in_trade = False
    side = Side.FLAT
    entry_bar = 0
    entry_fill = 0.0
    cooldown_until = 0

    for i in range(n):
        if i in session_set:
            if in_trade:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": i - entry_bar, "side": side, "exit": "session"})
                in_trade = False
                side = Side.FLAT
            cooldown_until = i + min_hold

        if in_trade:
            hold = i - entry_bar
            if side == Side.LONG:
                mid_pnl = (mid[i] - entry_fill) / MES_TICK
            else:
                mid_pnl = (entry_fill - mid[i]) / MES_TICK

            if mid_pnl >= tp_ticks:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": hold, "side": side, "exit": "tp"})
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue
            if mid_pnl <= -sl_ticks:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": hold, "side": side, "exit": "sl"})
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue
            if hold >= max_hold:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append({"pnl": pnl, "hold": hold, "side": side, "exit": "max_hold"})
                in_trade = False
                side = Side.FLAT
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
            elif z <= -entry_z:
                in_trade = True
                side = Side.SHORT
                entry_bar = i
                entry_fill = bid[i]

    if in_trade:
        pnl = _bracket_pnl(side, entry_fill, bid[n-1], ask[n-1])
        trades.append({"pnl": pnl, "hold": n - 1 - entry_bar, "side": side, "exit": "session"})

    if not trades:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    holds = np.array([t["hold"] for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    gp = winners.sum() if len(winners) > 0 else 0.0
    gl = abs(losers.sum()) if len(losers) > 0 else 0.0
    cum = np.cumsum(pnls)
    max_dd = (np.maximum.accumulate(cum) - cum).max()
    n_t = len(trades)
    n_long = sum(1 for t in trades if t["side"] == Side.LONG)

    return {
        "n": n_t, "n_long": n_long, "n_short": n_t - n_long,
        "wr": len(winners) / n_t, "avg_pnl": float(pnls.mean()),
        "pf": gp / gl if gl > 0 else float("inf"),
        "net": float(pnls.sum()) * MES_TICK_VALUE - n_t * commission_rt,
        "max_dd": float(max_dd), "avg_hold": float(holds.mean()),
    }


@click.command()
@click.option("--year", default=2026, type=int)
@click.option("--model-dir", default=None, type=str)
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
    year: int, model_dir: str | None, macro_hmm_path: str, micro_hmm_path: str,
    window_size: int, lookback: int, max_hold: int, min_hold: int,
    commission: float, batch_size: int, verbose: bool,
) -> None:
    """Sweep all regime models on their respective regimes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}_h5"
    else:
        model_dir_path = Path(model_dir)

    click.echo(f"Model dir: {model_dir_path}")
    click.echo(f"Year: {year}, Device: {device}")
    click.echo()

    # ── Load data once ────────────────────────────────────────────
    click.echo("[1/4] LOADING DATA...")
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
    click.echo(f"  {n_bars:,} bars")

    # ── Compute regime labels once ────────────────────────────────
    click.echo("\n[2/4] COMPUTING REGIME LABELS...")
    bars_for_macro = pd.DataFrame({
        "close": bars["mid"].values, "volume": bars["total_vol"].values,
        "timestamp": bars.index.values,
    })
    macro_feats, _ = macro_features_from_1s_bars(bars_for_macro, window=DEFAULT_MACRO_WINDOW)

    bars_for_micro = pd.DataFrame({
        "close": bars["mid"].values, "volume": bars["total_vol"].values,
        "spread_ticks": bars["spread_ticks"].values,
        "ofi": (np.diff(bars["bid_sz"].values, prepend=bars["bid_sz"].values[0])
                - np.diff(bars["ask_sz"].values, prepend=bars["ask_sz"].values[0])),
        "timestamp": bars.index.values,
    })
    micro_feats, _ = micro_features_from_1s_bars(bars_for_micro, window=DEFAULT_MICRO_WINDOW)
    del bars, bars_for_macro, bars_for_micro
    gc.collect()

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

    # Print regime distribution
    for m in range(macro_hmm.n_states):
        for u in range(micro_hmm.n_states):
            mask = (macro_per_bar == m) & (micro_per_bar == u)
            pct = mask.mean() * 100
            if pct > 0.1:
                click.echo(f"  Regime ({m},{u}): {mask.sum():,} bars ({pct:.1f}%)")

    # ── Run inference + sweep per model ───────────────────────────
    click.echo(f"\n[3/4] RUNNING SWEEPS PER REGIME MODEL...")

    entry_zs = [1.5, 2.0, 2.5, 3.0]
    tp_sls = [(4, 4), (6, 6), (8, 8), (9, 9), (8, 6), (9, 6), (12, 8)]

    all_best = []

    # Find all pair models
    model_files = sorted(model_dir_path.glob("pair_*.pt"))
    click.echo(f"  Found {len(model_files)} regime models")

    for model_path in model_files:
        model_name = model_path.stem  # e.g., "pair_3_0"
        parts = model_name.split("_")
        target_macro = int(parts[1])
        target_micro = int(parts[2])

        regime_mask = (macro_per_bar == target_macro) & (micro_per_bar == target_micro)
        regime_pct = regime_mask.mean() * 100

        if regime_mask.sum() < 100:
            click.echo(f"\n  {model_name}: regime ({target_macro},{target_micro}) has {regime_mask.sum()} bars — SKIP")
            continue

        click.echo(f"\n  {model_name}: regime ({target_macro},{target_micro}) = {regime_mask.sum():,} bars ({regime_pct:.1f}%)")

        # Load model and run inference
        model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        p_up = rolling_inference(model, features, window_size=window_size,
                                 batch_size=batch_size, device=device)
        z = _rolling_zscore(p_up, lookback=lookback)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Sweep
        best_net = float("-inf")
        best = None

        for entry_z in entry_zs:
            for tp, sl in tp_sls:
                r = _run_bracket_backtest(
                    z, mid, bid, ask, session_breaks,
                    entry_z=entry_z, tp_ticks=float(tp), sl_ticks=float(sl),
                    max_hold=max_hold, min_hold=min_hold,
                    commission_rt=commission, regime_mask=regime_mask,
                )
                if r is None or r["n"] < 5:
                    continue
                if r["net"] > best_net:
                    best_net = r["net"]
                    best = {**r, "model": model_name, "regime": f"{target_macro}_{target_micro}",
                            "entry_z": entry_z, "tp": tp, "sl": sl, "regime_pct": regime_pct}

        if best:
            all_best.append(best)
            click.echo(f"    BEST: z={best['entry_z']}, TP={best['tp']}, SL={best['sl']} -> "
                       f"{best['n']} trades, WR={best['wr']:.1%}, PF={best['pf']:.2f}, "
                       f"Net=${best['net']:+,.2f}, MaxDD={best['max_dd']:.0f}t, Hold={best['avg_hold']:.0f}s")
        else:
            click.echo(f"    No viable configs")

    # ── Summary ───────────────────────────────────────────────────
    click.echo(f"\n{'=' * 80}")
    click.echo("[4/4] COMBINED SUMMARY")
    click.echo(f"{'=' * 80}")

    if not all_best:
        click.echo("  No profitable configs found.")
        return

    click.echo(f"\n  {'Model':<12} | {'Regime':>6} | {'Reg%':>5} | {'EZ':>3} | {'TP':>2}:{' SL':<2} | "
               f"{'Trades':>6} | {'L':>4}/{' S':<4} | {'WR':>5} | {'PF':>5} | {'Net$':>10} | {'MaxDD':>5} | {'Hold':>5}")
    click.echo(f"  {'-'*12}-+-{'-'*6}-+-{'-'*5}-+-{'-'*3}-+-{'-'*5}-+-"
               f"{'-'*6}-+-{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}")

    total_net = 0
    total_trades = 0

    for b in sorted(all_best, key=lambda x: x["net"], reverse=True):
        click.echo(
            f"  {b['model']:<12} | {b['regime']:>6} | {b['regime_pct']:>4.1f}% | "
            f"{b['entry_z']:>3.1f} | {b['tp']:>2}:{b['sl']:<2} | "
            f"{b['n']:>6} | {b['n_long']:>4}/{b['n_short']:<4} | "
            f"{b['wr']:>5.1%} | {b['pf']:>5.2f} | {b['net']:>+10.2f} | "
            f"{b['max_dd']:>5.0f} | {b['avg_hold']:>5.0f}s"
        )
        total_net += b["net"]
        total_trades += b["n"]

    click.echo(f"\n  COMBINED: {total_trades} trades, Net=${total_net:+,.2f}")
    click.echo(f"  (Assumes independent regime models running simultaneously)")


if __name__ == "__main__":
    main()
