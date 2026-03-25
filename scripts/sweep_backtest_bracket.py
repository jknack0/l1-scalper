"""Sweep using z-score entry + fixed bracket exit.

Model picks entry timing/direction via z-score, then a fixed TP/SL bracket
resolves the trade. No signal-based exit — once you're in, the bracket
decides. This separates "when to enter" (model) from "how to exit" (bracket).

Usage:
    python scripts/sweep_backtest_bracket.py --year 2026 --model-name fallback
"""

from __future__ import annotations

import gc
import logging
import sys
import time
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
from src.regime.macro_features_v2 import DEFAULT_MACRO_WINDOW, MACRO_FEATURE_NAMES, macro_features_from_1s_bars
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_features_v2 import DEFAULT_MICRO_WINDOW, MICRO_FEATURE_NAMES, micro_features_from_1s_bars
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
L1_DIR = DATA_DIR / "l1"

logger = logging.getLogger(__name__)


def _rolling_zscore(p_up: np.ndarray, lookback: int = 300) -> np.ndarray:
    """Rolling z-score of P(up)."""
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


def _run_bracket_backtest(
    z_scores: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    session_breaks: np.ndarray,
    entry_z: float,
    tp_ticks: float,
    sl_ticks: float,
    max_hold: int,
    min_hold: int,
    commission_rt: float,
    regime_mask: np.ndarray | None = None,
) -> dict:
    """Backtest: z-score entry + bracket exit.

    Entry: z >= entry_z → long, z <= -entry_z → short (only when regime_mask[i] is True)
    Exit: first of TP, SL, max_hold, or session end.
    After exit, wait min_hold bars before next entry (cooldown).
    """
    n = len(z_scores)
    session_set = set(session_breaks) if session_breaks is not None else set()

    trades: list[Trade] = []
    in_trade = False
    side = Side.FLAT
    entry_bar = 0
    entry_fill = 0.0
    cooldown_until = 0

    for i in range(n):
        # Session boundary: force close
        if i in session_set:
            if in_trade:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_fill, exit_price=mid[i],
                    entry_p_up=0.0, exit_p_up=0.0,
                    exit_reason="session_end", pnl_ticks=pnl,
                    hold_bars=i - entry_bar,
                ))
                in_trade = False
                side = Side.FLAT
            cooldown_until = i + min_hold

        if in_trade:
            hold = i - entry_bar

            # Check TP/SL
            if side == Side.LONG:
                # TP: can we sell at bid >= entry + tp?
                profit_ticks = (bid[i] - entry_fill) / MES_TICK
                loss_ticks = (entry_fill - ask[i]) / MES_TICK  # worst case if we had to buy back at ask... no
                # Simpler: track mid-based P&L for TP/SL, use bid/ask for actual fill
                mid_pnl = (mid[i] - entry_fill) / MES_TICK
            else:
                mid_pnl = (entry_fill - mid[i]) / MES_TICK

            if mid_pnl >= tp_ticks:
                # TP hit — fill at bid (long) or ask (short)
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_fill, exit_price=mid[i],
                    entry_p_up=0.0, exit_p_up=0.0,
                    exit_reason="tp", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue

            if mid_pnl <= -sl_ticks:
                # SL hit
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_fill, exit_price=mid[i],
                    entry_p_up=0.0, exit_p_up=0.0,
                    exit_reason="sl", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue

            if hold >= max_hold:
                pnl = _bracket_pnl(side, entry_fill, bid[i], ask[i])
                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i, side=side,
                    entry_price=entry_fill, exit_price=mid[i],
                    entry_p_up=0.0, exit_p_up=0.0,
                    exit_reason="max_hold", pnl_ticks=pnl,
                    hold_bars=hold,
                ))
                in_trade = False
                side = Side.FLAT
                cooldown_until = i + min_hold
                continue

        else:
            # Check entry
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
                entry_fill = ask[i]  # cross spread to buy
            elif z <= -entry_z:
                in_trade = True
                side = Side.SHORT
                entry_bar = i
                entry_fill = bid[i]  # cross spread to sell

    # Force close remaining
    if in_trade:
        pnl = _bracket_pnl(side, entry_fill, bid[n-1], ask[n-1])
        trades.append(Trade(
            entry_bar=entry_bar, exit_bar=n-1, side=side,
            entry_price=entry_fill, exit_price=mid[n-1],
            entry_p_up=0.0, exit_p_up=0.0,
            exit_reason="session_end", pnl_ticks=pnl,
            hold_bars=n - 1 - entry_bar,
        ))

    return _summarize(trades, commission_rt)


def _bracket_pnl(side: Side, entry_fill: float, bid: float, ask: float) -> float:
    """P&L with realistic fills."""
    if side == Side.LONG:
        return (bid - entry_fill) / MES_TICK  # sell at bid
    else:
        return (entry_fill - ask) / MES_TICK  # buy back at ask


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
        "exits_tp": reasons.count("tp"),
        "exits_sl": reasons.count("sl"),
        "exits_hold": reasons.count("max_hold"),
        "exits_session": reasons.count("session_end"),
    }


@click.command()
@click.option("--year", default=2026, type=int)
@click.option("--model-dir", default=None, type=str)
@click.option("--model-name", default="fallback", type=str)
@click.option("--window-size", default=30, type=int)
@click.option("--lookback", default=300, type=int)
@click.option("--max-hold", default=300, type=int)
@click.option("--min-hold", default=30, type=int, help="Cooldown after trade in bars.")
@click.option("--commission", default=0.59, type=float)
@click.option("--batch-size", default=4096, type=int)
@click.option("--macro-hmm-path", default=None, type=str,
              help="Macro HMM for regime gating. If set with --regime, only enter during that regime.")
@click.option("--micro-hmm-path", default=None, type=str, help="Micro HMM for regime gating.")
@click.option("--regime", default=None, type=str,
              help="Regime pair to gate on, e.g., '3_0'. Requires --macro-hmm-path and --micro-hmm-path.")
@click.option("--verbose", "-v", is_flag=True)
def main(
    year: int,
    model_dir: str | None,
    model_name: str,
    window_size: int,
    lookback: int,
    max_hold: int,
    min_hold: int,
    commission: float,
    batch_size: int,
    macro_hmm_path: str | None,
    micro_hmm_path: str | None,
    regime: str | None,
    verbose: bool,
) -> None:
    """Sweep z-score entry + bracket exit combinations."""
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
    click.echo(f"Year: {year}, Lookback: {lookback}s, Min hold: {min_hold}s")
    if regime:
        click.echo(f"Regime gate: {regime}")
    click.echo()

    # ── Load + inference ──────────────────────────────────────────
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

    # ── Regime mask (optional) ────────────────────────────────────
    regime_mask = None
    if regime and macro_hmm_path and micro_hmm_path:
        click.echo(f"  Computing regime labels...")
        target_macro, target_micro = int(regime.split("_")[0]), int(regime.split("_")[1])

        # Compute macro features from bars (need close + volume + timestamp)
        import pandas as pd
        bars_for_macro = pd.DataFrame({
            "close": bars["mid"].values,
            "volume": bars["total_vol"].values if "total_vol" in bars.columns else np.ones(n_bars),
            "timestamp": bars.index.values,
        })
        macro_feats, _ = macro_features_from_1s_bars(bars_for_macro, window=DEFAULT_MACRO_WINDOW)

        # Compute micro features (need close, volume, spread_ticks, ofi)
        bars_for_micro = pd.DataFrame({
            "close": bars["mid"].values,
            "volume": bars["total_vol"].values if "total_vol" in bars.columns else np.ones(n_bars),
            "spread_ticks": bars["spread_ticks"].values if "spread_ticks" in bars.columns else np.ones(n_bars),
            "ofi": (np.diff(bars["bid_sz"].values, prepend=bars["bid_sz"].values[0])
                    - np.diff(bars["ask_sz"].values, prepend=bars["ask_sz"].values[0]))
                   if "bid_sz" in bars.columns else np.zeros(n_bars),
            "timestamp": bars.index.values,
        })
        micro_feats, _ = micro_features_from_1s_bars(bars_for_micro, window=DEFAULT_MICRO_WINDOW)

        # Load HMMs and run forward-only filtering
        macro_hmm = MacroRegimeHMMv2()
        macro_hmm.load(macro_hmm_path)
        micro_hmm = MicroRegimeHMMv2()
        micro_hmm.load(micro_hmm_path)

        macro_norm = macro_hmm.normalize(macro_feats)
        micro_norm = micro_hmm.normalize(micro_feats)

        macro_posteriors = macro_hmm.predict_proba_forward(macro_norm)
        micro_posteriors = micro_hmm.predict_proba_forward(micro_norm)

        macro_labels = macro_posteriors.argmax(axis=1)
        micro_labels = micro_posteriors.argmax(axis=1)

        # Expand to 1-sec bar resolution
        macro_per_bar = np.repeat(macro_labels, DEFAULT_MACRO_WINDOW)[:n_bars]
        micro_per_bar = np.repeat(micro_labels, DEFAULT_MICRO_WINDOW)[:n_bars]

        # Pad if shorter than n_bars
        if len(macro_per_bar) < n_bars:
            macro_per_bar = np.pad(macro_per_bar, (0, n_bars - len(macro_per_bar)),
                                   constant_values=-1)
        if len(micro_per_bar) < n_bars:
            micro_per_bar = np.pad(micro_per_bar, (0, n_bars - len(micro_per_bar)),
                                   constant_values=-1)

        regime_mask = (macro_per_bar == target_macro) & (micro_per_bar == target_micro)
        pct = regime_mask.mean() * 100
        click.echo(f"  Regime ({target_macro},{target_micro}): {regime_mask.sum():,} bars ({pct:.1f}%)")

    del bars
    gc.collect()

    click.echo(f"  {len(features):,} bars")

    click.echo("\n[2/3] ROLLING INFERENCE...")
    model = EntryModel(n_features=features.shape[1], seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    p_up = rolling_inference(model, features, window_size=window_size,
                             batch_size=batch_size, device=device)

    del model, features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    z = _rolling_zscore(p_up, lookback=lookback)
    valid_z = z[np.isfinite(z)]
    click.echo(f"  Z-score: mean={valid_z.mean():.3f}, std={valid_z.std():.3f}")

    # ── Sweep ─────────────────────────────────────────────────────
    click.echo(f"\n[3/3] SWEEPING ENTRY Z + BRACKET PARAMS...")

    entry_zs = [1.0, 1.5, 2.0, 2.5, 3.0]
    tp_sls = [(4, 4), (6, 6), (8, 8), (9, 9), (6, 4), (8, 6), (9, 6), (12, 8)]

    click.echo(f"\n  {'EZ':>4} | {'TP':>3} | {'SL':>3} | {'Trades':>7} | "
               f"{'L':>5} | {'S':>6} | {'WR':>5} | {'AvgPnL':>7} | "
               f"{'PF':>5} | {'Net$':>10} | {'MaxDD':>6} | {'Hold':>5} | "
               f"{'TP%':>4} | {'SL%':>4} | {'MH%':>4}")
    click.echo(f"  {'-'*4}-+-{'-'*3}-+-{'-'*3}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-"
               f"{'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*5}-+-"
               f"{'-'*4}-+-{'-'*4}-+-{'-'*4}")

    best_net = float("-inf")
    best_params = None

    for entry_z_thresh in entry_zs:
        for tp, sl in tp_sls:
            r = _run_bracket_backtest(
                z, mid, bid, ask, session_breaks,
                entry_z=entry_z_thresh,
                tp_ticks=float(tp),
                sl_ticks=float(sl),
                max_hold=max_hold,
                min_hold=min_hold,
                commission_rt=commission,
                regime_mask=regime_mask,
            )

            if r["n_trades"] == 0:
                continue

            n = r["n_trades"]
            tp_pct = r["exits_tp"] / n * 100
            sl_pct = r["exits_sl"] / n * 100
            mh_pct = r["exits_hold"] / n * 100

            click.echo(
                f"  {entry_z_thresh:>4.1f} | {tp:>3} | {sl:>3} | "
                f"{n:>7,} | {r['n_long']:>5,} | {r['n_short']:>6,} | "
                f"{r['win_rate']:>5.1%} | {r['avg_pnl']:>+7.2f} | "
                f"{r['pf']:>5.2f} | {r['net_dollars']:>+10.2f} | "
                f"{r['max_dd']:>6.0f} | {r['avg_hold']:>5.0f}s | "
                f"{tp_pct:>4.0f} | {sl_pct:>4.0f} | {mh_pct:>4.0f}"
            )

            if r["net_dollars"] > best_net:
                best_net = r["net_dollars"]
                best_params = {**r, "entry_z": entry_z_thresh, "tp": tp, "sl": sl}

    if best_params:
        click.echo(f"\n  BEST: entry_z={best_params['entry_z']:.1f}, "
                   f"TP={best_params['tp']}, SL={best_params['sl']}")
        click.echo(f"    {best_params['n_trades']} trades, WR={best_params['win_rate']:.1%}, "
                   f"PF={best_params['pf']:.2f}, Net=${best_params['net_dollars']:+,.2f}, "
                   f"MaxDD={best_params['max_dd']:.0f}t, Hold={best_params['avg_hold']:.0f}s")


if __name__ == "__main__":
    main()
