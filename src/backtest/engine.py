"""Backtest engine — runs rolling inference + position manager on historical data.

Takes precomputed 1-sec bar features + mid prices, runs the model every second,
and simulates trades with realistic bid/ask fills.

Fill model:
    - Long entry: fill at ask (cross spread to buy)
    - Short entry: fill at bid (cross spread to sell)
    - Long exit: fill at bid (cross spread to sell)
    - Short exit: fill at ask (cross spread to buy)
    - This means ~1 tick round-trip spread cost on entries + exits

Output: list of Trade objects with P&L, hold times, and exit reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.backtest.position_manager import (
    MES_TICK,
    MES_TICK_VALUE,
    PositionManager,
    PositionManagerConfig,
    AdaptiveStopConfig,
    Side,
    Trade,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary statistics from a backtest run."""
    trades: list[Trade]
    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    total_pnl_ticks: float
    total_pnl_dollars: float
    avg_pnl_ticks: float
    avg_winner_ticks: float
    avg_loser_ticks: float
    profit_factor: float
    max_drawdown_ticks: float
    avg_hold_bars: float
    commission_total: float
    net_pnl_dollars: float

    # Exit reason breakdown
    exits_signal: int
    exits_hard_sl: int
    exits_trail: int
    exits_breakeven: int
    exits_tier1: int
    exits_tier2: int
    exits_tier3: int
    exits_velocity: int
    exits_max_hold: int
    exits_session_end: int


def run_backtest(
    p_up: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    session_breaks: np.ndarray | None = None,
    config: PositionManagerConfig | AdaptiveStopConfig | None = None,
) -> BacktestResult:
    """Run backtest on P(up) signal with position management.

    Args:
        p_up: [n_bars] P(up) from rolling inference.
        mid: [n_bars] mid prices.
        bid: [n_bars] bid prices (for realistic fills). If None, uses mid.
        ask: [n_bars] ask prices. If None, uses mid.
        session_breaks: bar indices where new sessions start (force close).
        config: position manager configuration.

    Returns:
        BacktestResult with trade list and summary stats.
    """
    config = config or PositionManagerConfig()
    pm = PositionManager(config)
    n_bars = len(p_up)

    session_break_set = set(session_breaks) if session_breaks is not None else set()

    for i in range(n_bars):
        # Force close at session boundaries
        if i in session_break_set and not pm.is_flat:
            pm.force_close(i, p_up[i] if np.isfinite(p_up[i]) else 0.5, mid[i])
            pm.reset()

        pm.update(i, p_up[i], mid[i])

    # Force close any remaining position
    if not pm.is_flat:
        last_valid = n_bars - 1
        pm.force_close(last_valid, p_up[last_valid] if np.isfinite(p_up[last_valid]) else 0.5, mid[last_valid])

    trades = pm.trades

    # Apply realistic fill adjustment: entry at ask (long) / bid (short),
    # exit at bid (long) / ask (short)
    if bid is not None and ask is not None:
        for t in trades:
            if t.side == Side.LONG:
                # Entered at ask, exited at bid
                fill_entry = ask[t.entry_bar]
                fill_exit = bid[t.exit_bar]
                t.pnl_ticks = (fill_exit - fill_entry) / MES_TICK
            else:
                # Entered at bid, exited at ask
                fill_entry = bid[t.entry_bar]
                fill_exit = ask[t.exit_bar]
                t.pnl_ticks = (fill_entry - fill_exit) / MES_TICK

    return _compute_stats(trades, config.commission_rt_dollars)


def _compute_stats(trades: list[Trade], commission_rt: float) -> BacktestResult:
    """Compute summary statistics from trade list."""
    n = len(trades)

    if n == 0:
        return BacktestResult(
            trades=trades, n_trades=0, n_long=0, n_short=0,
            win_rate=0.0, total_pnl_ticks=0.0, total_pnl_dollars=0.0,
            avg_pnl_ticks=0.0, avg_winner_ticks=0.0, avg_loser_ticks=0.0,
            profit_factor=0.0, max_drawdown_ticks=0.0, avg_hold_bars=0.0,
            commission_total=0.0, net_pnl_dollars=0.0,
            exits_signal=0, exits_hard_sl=0, exits_trail=0,
            exits_breakeven=0, exits_tier1=0, exits_tier2=0, exits_tier3=0, exits_velocity=0,
            exits_max_hold=0, exits_session_end=0,
        )

    pnls = np.array([t.pnl_ticks for t in trades])
    holds = np.array([t.hold_bars for t in trades])

    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    # Profit factor
    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (in ticks)
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    # Commission
    commission_total = n * commission_rt
    total_pnl_dollars = float(pnls.sum()) * MES_TICK_VALUE
    net_pnl = total_pnl_dollars - commission_total

    # Exit reasons
    reasons = [t.exit_reason for t in trades]

    return BacktestResult(
        trades=trades,
        n_trades=n,
        n_long=sum(1 for t in trades if t.side == Side.LONG),
        n_short=sum(1 for t in trades if t.side == Side.SHORT),
        win_rate=float(len(winners) / n) if n > 0 else 0.0,
        total_pnl_ticks=float(pnls.sum()),
        total_pnl_dollars=total_pnl_dollars,
        avg_pnl_ticks=float(pnls.mean()),
        avg_winner_ticks=float(winners.mean()) if len(winners) > 0 else 0.0,
        avg_loser_ticks=float(losers.mean()) if len(losers) > 0 else 0.0,
        profit_factor=pf,
        max_drawdown_ticks=float(max_dd),
        avg_hold_bars=float(holds.mean()),
        commission_total=commission_total,
        net_pnl_dollars=net_pnl,
        exits_signal=reasons.count("signal"),
        exits_hard_sl=reasons.count("hard_sl"),
        exits_trail=reasons.count("trail"),
        exits_breakeven=reasons.count("breakeven"),
        exits_tier1=reasons.count("tier1"),
        exits_tier2=reasons.count("tier2"),
        exits_tier3=reasons.count("tier3"),
        exits_velocity=reasons.count("velocity"),
        exits_max_hold=reasons.count("max_hold"),
        exits_session_end=reasons.count("session_end"),
    )


def print_result(result: BacktestResult) -> None:
    """Print backtest results to stdout."""
    r = result
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Trades:          {r.n_trades:,} ({r.n_long} long, {r.n_short} short)")
    print(f"  Win rate:        {r.win_rate:.1%}")
    print(f"  Avg P&L:         {r.avg_pnl_ticks:+.2f} ticks/trade")
    print(f"  Avg winner:      {r.avg_winner_ticks:+.2f} ticks")
    print(f"  Avg loser:       {r.avg_loser_ticks:+.2f} ticks")
    print(f"  Profit factor:   {r.profit_factor:.2f}")
    print(f"  Total P&L:       {r.total_pnl_ticks:+.1f} ticks (${r.total_pnl_dollars:+,.2f})")
    print(f"  Commission:      ${r.commission_total:,.2f}")
    print(f"  Net P&L:         ${r.net_pnl_dollars:+,.2f}")
    print(f"  Max drawdown:    {r.max_drawdown_ticks:.1f} ticks (${r.max_drawdown_ticks * MES_TICK_VALUE:,.2f})")
    print(f"  Avg hold:        {r.avg_hold_bars:.0f} bars ({r.avg_hold_bars:.0f}s)")
    print(f"  Exits:           signal={r.exits_signal}, hard_sl={r.exits_hard_sl}, "
          f"trail={r.exits_trail}, max_hold={r.exits_max_hold}, session={r.exits_session_end}")
    if any([r.exits_breakeven, r.exits_tier1, r.exits_tier2, r.exits_tier3, r.exits_velocity]):
        print(f"  Adaptive exits:  breakeven={r.exits_breakeven}, tier1={r.exits_tier1}, "
              f"tier2={r.exits_tier2}, tier3={r.exits_tier3}, velocity={r.exits_velocity}")
