"""Sweep using rolling z-score of P(up) instead of absolute thresholds.

Computes a rolling z-score of P(up) over a lookback window, then enters
when the z-score exceeds thresholds. This adapts to the model's actual
output distribution instead of assuming calibrated probabilities.

Usage:
    python scripts/sweep_backtest_zscore.py --year 2026 --model-name fallback
    python scripts/sweep_backtest_zscore.py --year 2026 --model-name pair_3_0
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.position_manager import MES_TICK, MES_TICK_VALUE, Side, Trade
from src.backtest.rolling_inference import rolling_inference
from src.models.dataset import _compute_features, _filter_rth, _resample_to_1sec, _z_score_normalize
from src.models.entry_model import EntryModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)


def _rolling_zscore(p_up: np.ndarray, lookback: int = 300) -> np.ndarray:
    """Compute rolling z-score of P(up) using expanding then rolling window.

    Uses Welford's-style cumsum trick for O(n) computation.
    Returns NaN for bars without enough history.
    """
    n = len(p_up)
    z = np.full(n, np.nan, dtype=np.float32)

    # Cumulative sum and sum-of-squares for rolling stats
    valid = np.isfinite(p_up)
    p_clean = np.where(valid, p_up, 0.0)

    cs = np.cumsum(p_clean)
    cs2 = np.cumsum(p_clean ** 2)
    cv = np.cumsum(valid.astype(np.float64))

    for i in range(lookback, n):
        if not valid[i]:
            continue

        # Rolling window stats
        j = i - lookback
        count = cv[i] - cv[j]
        if count < 30:  # need minimum samples
            continue

        s = cs[i] - cs[j]
        s2 = cs2[i] - cs2[j]
        mean = s / count
        var = s2 / count - mean ** 2
        if var < 1e-10:
            continue

        std = np.sqrt(var)
        z[i] = (p_up[i] - mean) / std

    return z


def _run_zscore_backtest(
    z_scores: np.ndarray,
    p_up: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
    long_entry_z: float,
    short_entry_z: float,
    exit_z: float,
    hard_sl_ticks: float,
    max_hold_bars: int,
    commission_rt: float,
) -> dict:
    """Run backtest using z-score thresholds.

    Entry: z >= long_entry_z → long, z <= short_entry_z → short
    Exit: z crosses back through exit_z (toward 0)
    """
    n = len(z_scores)
    session_set = set(session_breaks) if session_breaks is not None else set()

    trades: list[Trade] = []
    side = Side.FLAT
    entry_bar = 0
    entry_price = 0.0
    entry_z = 0.0
    entry_p = 0.0

    for i in range(n):
        z = z_scores[i]

        # Session boundary: force close
        if i in session_set and side != Side.FLAT:
            pnl = _calc_pnl(side, entry_price, mid[i], bid, ask, entry_bar, i)
            trades.append(Trade(
                entry_bar=entry_bar, exit_bar=i, side=side,
                entry_price=entry_price, exit_price=mid[i],
                entry_p_up=entry_p, exit_p_up=p_up[i] if np.isfinite(p_up[i]) else 0.5,
                exit_reason="session_end", pnl_ticks=pnl,
                hold_bars=i - entry_bar,
            ))
            side = Side.FLAT

        if np.isnan(z):
            continue

        # Check exits
        if side != Side.FLAT:
            hold = i - entry_bar

            # Hard SL
            if side == Side.LONG:
                adverse = (entry_price - mid[i]) / MES_TICK
            else:
                adverse = (mid[i] - entry_price) / MES_TICK

            if adverse >= hard_sl_ticks:
                pnl = _calc_pnl(side, entry_price, mid[i], bid, ask, entry_bar, i)
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_price, exit_price=mid[i],
                    entry_p_up=entry_p, exit_p_up=p_up[i],
                    exit_reason="hard_sl", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                side = Side.FLAT
                continue

            # Max hold
            if hold >= max_hold_bars:
                pnl = _calc_pnl(side, entry_price, mid[i], bid, ask, entry_bar, i)
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_price, exit_price=mid[i],
                    entry_p_up=entry_p, exit_p_up=p_up[i],
                    exit_reason="max_hold", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                side = Side.FLAT
                continue

            # Z-score signal exit
            if side == Side.LONG and z < exit_z:
                pnl = _calc_pnl(side, entry_price, mid[i], bid, ask, entry_bar, i)
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_price, exit_price=mid[i],
                    entry_p_up=entry_p, exit_p_up=p_up[i],
                    exit_reason="signal", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                side = Side.FLAT
            elif side == Side.SHORT and z > -exit_z:
                pnl = _calc_pnl(side, entry_price, mid[i], bid, ask, entry_bar, i)
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_price, exit_price=mid[i],
                    entry_p_up=entry_p, exit_p_up=p_up[i],
                    exit_reason="signal", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                side = Side.FLAT

        # Check entries (only if flat)
        if side == Side.FLAT:
            if z >= long_entry_z:
                side = Side.LONG
                entry_bar = i
                entry_price = mid[i]
                entry_z = z
                entry_p = p_up[i]
            elif z <= short_entry_z:
                side = Side.SHORT
                entry_bar = i
                entry_price = mid[i]
                entry_z = z
                entry_p = p_up[i]

    # Force close remaining
    if side != Side.FLAT:
        pnl = _calc_pnl(side, entry_price, mid[n-1], bid, ask, entry_bar, n-1)
        trades.append(Trade(
            entry_bar=entry_bar, exit_bar=n-1, side=side,
            entry_price=entry_price, exit_price=mid[n-1],
            entry_p_up=entry_p, exit_p_up=0.5,
            exit_reason="session_end", pnl_ticks=pnl,
            hold_bars=n - 1 - entry_bar,
        ))

    return _summarize(trades, commission_rt)


def _calc_pnl(side: Side, entry_price: float, exit_mid: float,
              bid: np.ndarray, ask: np.ndarray, entry_bar: int, exit_bar: int) -> float:
    """Calculate P&L with realistic fills."""
    if side == Side.LONG:
        fill_entry = ask[entry_bar]
        fill_exit = bid[exit_bar]
        return (fill_exit - fill_entry) / MES_TICK
    else:
        fill_entry = bid[entry_bar]
        fill_exit = ask[exit_bar]
        return (fill_entry - fill_exit) / MES_TICK


def _summarize(trades: list[Trade], commission_rt: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}

    pnls = np.array([t.pnl_ticks for t in trades])
    holds = np.array([t.hold_bars for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = (running_max - cumulative).max()

    total_pnl_dollars = float(pnls.sum()) * MES_TICK_VALUE
    commission_total = n * commission_rt

    reasons = [t.exit_reason for t in trades]

    return {
        "n_trades": n,
        "n_long": sum(1 for t in trades if t.side == Side.LONG),
        "n_short": sum(1 for t in trades if t.side == Side.SHORT),
        "win_rate": float(len(winners) / n),
        "avg_pnl": float(pnls.mean()),
        "avg_win": float(winners.mean()) if len(winners) > 0 else 0.0,
        "avg_loss": float(losers.mean()) if len(losers) > 0 else 0.0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "total_ticks": float(pnls.sum()),
        "net_dollars": total_pnl_dollars - commission_total,
        "max_dd": float(max_dd),
        "avg_hold": float(holds.mean()),
        "exits_signal": reasons.count("signal"),
        "exits_sl": reasons.count("hard_sl"),
        "exits_hold": reasons.count("max_hold"),
        "exits_session": reasons.count("session_end"),
    }


@click.command()
@click.option("--year", default=2026, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--model-name", default="fallback", type=str)
@click.option("--window-size", default=30, type=int)
@click.option("--lookback", default=300, type=int, help="Z-score lookback window in bars.")
@click.option("--hard-sl", default=12.0, type=float)
@click.option("--max-hold", default=300, type=int)
@click.option("--commission", default=0.59, type=float)
@click.option("--batch-size", default=4096, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    year: int,
    model_dir: str | None,
    model_name: str,
    window_size: int,
    lookback: int,
    hard_sl: float,
    max_hold: int,
    commission: float,
    batch_size: int,
    verbose: bool,
) -> None:
    """Sweep z-score thresholds on rolling P(up) signal."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir is None:
        model_dir_path = MODEL_DIR / "regime_v2_fold2" / f"w{window_size}"
    else:
        model_dir_path = Path(model_dir)

    model_path = model_dir_path / f"{model_name}.pt"
    if not model_path.exists():
        click.echo(f"ERROR: Model not found at {model_path}")
        return

    click.echo(f"Model: {model_path}")
    click.echo(f"Year: {year}, Lookback: {lookback}s, Device: {device}")
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
    del bars
    gc.collect()

    click.echo(f"  {len(features):,} bars, {len(session_breaks)} sessions")

    # ── Inference ─────────────────────────────────────────────────
    click.echo("\n[2/3] ROLLING INFERENCE...")
    model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    t0 = time.time()
    p_up = rolling_inference(model, features, window_size=window_size,
                             batch_size=batch_size, device=device)
    click.echo(f"  Inference: {time.time() - t0:.1f}s")

    valid_p = p_up[np.isfinite(p_up)]
    click.echo(f"  P(up) raw: mean={valid_p.mean():.4f}, std={valid_p.std():.4f}")

    del model, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Z-score ───────────────────────────────────────────────────
    click.echo(f"\n  Computing rolling z-score (lookback={lookback})...")
    z = _rolling_zscore(p_up, lookback=lookback)
    valid_z = z[np.isfinite(z)]
    click.echo(f"  Z-score: mean={valid_z.mean():.4f}, std={valid_z.std():.4f}")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        click.echo(f"    p{pct:2d} = {np.percentile(valid_z, pct):+.3f}")

    # ── Sweep ─────────────────────────────────────────────────────
    click.echo(f"\n[3/3] SWEEPING Z-SCORE THRESHOLDS...")

    entry_zs = [1.0, 1.5, 2.0, 2.5, 3.0]
    exit_zs = [0.0, 0.25, 0.5, 0.75, 1.0]

    click.echo(f"\n  {'EntryZ':>6} | {'ExitZ':>5} | {'Trades':>7} | "
               f"{'Long':>5} | {'Short':>6} | {'WR':>5} | {'AvgPnL':>7} | "
               f"{'PF':>5} | {'Net$':>10} | {'MaxDD':>6} | {'AvgHold':>7} | {'SL':>3} | {'MaxH':>4}")
    click.echo(f"  {'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-"
               f"{'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}-+-{'-'*4}")

    best_net = float("-inf")
    best_params = None

    for entry_z_thresh in entry_zs:
        for exit_z_thresh in exit_zs:
            if exit_z_thresh >= entry_z_thresh:
                continue

            r = _run_zscore_backtest(
                z, p_up, mid, bid, ask, session_breaks,
                long_entry_z=entry_z_thresh,
                short_entry_z=-entry_z_thresh,
                exit_z=exit_z_thresh,
                hard_sl_ticks=hard_sl,
                max_hold_bars=max_hold,
                commission_rt=commission,
            )

            if r["n_trades"] == 0:
                continue

            click.echo(
                f"  {entry_z_thresh:>6.1f} | {exit_z_thresh:>5.2f} | "
                f"{r['n_trades']:>7,} | {r['n_long']:>5,} | {r['n_short']:>6,} | "
                f"{r['win_rate']:>5.1%} | {r['avg_pnl']:>+7.2f} | "
                f"{r['pf']:>5.2f} | {r['net_dollars']:>+10.2f} | "
                f"{r['max_dd']:>6.0f} | {r['avg_hold']:>7.0f}s | "
                f"{r['exits_sl']:>3} | {r['exits_hold']:>4}"
            )

            if r["net_dollars"] > best_net:
                best_net = r["net_dollars"]
                best_params = {**r, "entry_z": entry_z_thresh, "exit_z": exit_z_thresh}

    if best_params:
        click.echo(f"\n  BEST: entry_z={best_params['entry_z']:.1f}, exit_z={best_params['exit_z']:.2f}")
        click.echo(f"    {best_params['n_trades']} trades ({best_params['n_long']}L/{best_params['n_short']}S), "
                   f"WR={best_params['win_rate']:.1%}, PF={best_params['pf']:.2f}, "
                   f"Net=${best_params['net_dollars']:+,.2f}, "
                   f"MaxDD={best_params['max_dd']:.0f}t, Hold={best_params['avg_hold']:.0f}s")
    else:
        click.echo("\n  No trades generated.")


if __name__ == "__main__":
    main()
