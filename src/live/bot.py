"""Live paper trading bot.

Connects to Databento live feed, processes ticks through the streaming
feature pipeline, runs regime-gated model inference, and executes
trades via Tradovate with trailing stop exits.

Architecture:
    Databento live ticks
      → StreamingFeatures (1-sec bars + 15 features + z-score)
      → Model inference every second (P(up))
      → Rolling z-score of P(up) for entry signals
      → TrailingStopManager for exit management
      → TradovateExecutor for order submission
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import torch

from src.config.settings import DatabentoSettings
from src.live.streaming_features import StreamingFeatures
from src.live.tradovate_executor import TradovateExecutor
from src.models.entry_model import EntryModel
from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MES_TICK = 0.25


class Side(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class StreamingRegimeDetector:
    """Accumulates 1-sec bar data into macro/micro windows and runs HMMs.

    Macro: 300-bar (5-min) windows → 7 features → forward filter
    Micro: 30-bar (30-sec) windows → 6 features → forward filter
    """

    def __init__(
        self,
        macro_hmm_path: str = "models/macro_hmm_v2.pkl",
        micro_hmm_path: str = "models/micro_hmm_v2.pkl",
    ) -> None:
        self.macro_hmm = MacroRegimeHMMv2()
        self.macro_hmm.load(macro_hmm_path)
        self.micro_hmm = MicroRegimeHMMv2()
        self.micro_hmm.load(micro_hmm_path)

        self._macro_window = 300
        self._micro_window = 30

        # Accumulate bar-level data for feature computation
        self._log_returns: list[float] = []
        self._volumes: list[float] = []
        self._spreads: list[float] = []
        self._ofis: list[float] = []

        self._macro_state: int = -1
        self._micro_state: int = -1
        self._bars_seen: int = 0

    def on_bar(self, log_ret: float, volume: float, spread_ticks: float, ofi: float) -> None:
        """Feed one 1-sec bar's data. Updates regime state when windows complete."""
        self._log_returns.append(log_ret)
        self._volumes.append(volume)
        self._spreads.append(spread_ticks)
        self._ofis.append(ofi)
        self._bars_seen += 1

        # Micro update every 30 bars
        if self._bars_seen % self._micro_window == 0 and self._bars_seen >= self._micro_window:
            self._update_micro()

        # Macro update every 300 bars
        if self._bars_seen % self._macro_window == 0 and self._bars_seen >= self._macro_window:
            self._update_macro()

    def _update_macro(self) -> None:
        """Compute macro features from last 300 bars and run HMM."""
        rets = np.array(self._log_returns[-self._macro_window:])
        vols = np.array(self._volumes[-self._macro_window:])

        # 7 macro features (matching macro_features_v2.py)
        realized_vol = rets.std()
        # Vol of vol (rolling 60-bar std of vol)
        if len(rets) >= 120:
            rolling_vols = [rets[i:i+60].std() for i in range(0, len(rets)-60, 10)]
            vol_of_vol = np.std(rolling_vols) if rolling_vols else 0.0
        else:
            vol_of_vol = 0.0
        # Autocorr
        if len(rets) > 2:
            r0, r1 = rets[:-1], rets[1:]
            d0, d1 = r0 - r0.mean(), r1 - r1.mean()
            denom = np.sqrt((d0**2).sum() * (d1**2).sum())
            autocorr = float((d0 * d1).sum() / denom) if denom > 1e-15 else 0.0
        else:
            autocorr = 0.0
        # Variance ratio
        var1 = rets.var(ddof=1) if len(rets) > 1 else 1e-10
        k = 5
        if len(rets) >= k * 2 and var1 > 1e-15:
            k_rets = np.convolve(rets, np.ones(k), mode="valid")
            var_ratio = k_rets.var(ddof=1) / (k * var1)
        else:
            var_ratio = 1.0
        # Efficiency ratio
        total_path = np.abs(rets).sum()
        efficiency = abs(rets.sum()) / total_path if total_path > 1e-15 else 0.0
        # Volume kurtosis
        m = vols.mean()
        diff = vols - m
        m2 = (diff**2).mean()
        m4 = (diff**4).mean()
        kurtosis = (m4 / (m2**2) - 3.0) if m2 > 1e-30 else 0.0
        # Return skew
        m_r = rets.mean()
        diff_r = rets - m_r
        m2_r = (diff_r**2).mean()
        m3_r = (diff_r**3).mean()
        skew = (m3_r / (m2_r**1.5)) if m2_r > 1e-30 else 0.0

        feats = np.array([[realized_vol, vol_of_vol, autocorr, var_ratio,
                           efficiency, kurtosis, skew]])
        norm = self.macro_hmm.normalize(feats)
        posterior = self.macro_hmm.predict_proba_incremental(norm[0])
        self._macro_state = int(np.argmax(posterior))
        logger.info("Macro regime: state=%d (conf=%.2f)", self._macro_state, posterior.max())

    def _update_micro(self) -> None:
        """Compute micro features from last 30 bars and run HMM."""
        rets = np.array(self._log_returns[-self._micro_window:])
        spreads = np.array(self._spreads[-self._micro_window:])
        volumes = np.array(self._volumes[-self._micro_window:])
        ofis = np.array(self._ofis[-self._micro_window:])

        # 6 micro features (matching micro_features_v2.py)
        spread_mean = spreads.mean()
        spread_cv = spreads.std() / spread_mean if spread_mean > 1e-10 else 0.0
        trade_rate = volumes.mean()
        total_vol = volumes.sum()
        ofi_momentum = ofis.sum() / total_vol if total_vol > 0 else 0.0
        # Autocorr
        if len(rets) > 2:
            r0, r1 = rets[:-1], rets[1:]
            d0, d1 = r0 - r0.mean(), r1 - r1.mean()
            denom = np.sqrt((d0**2).sum() * (d1**2).sum())
            autocorr = float((d0 * d1).sum() / denom) if denom > 1e-15 else 0.0
        else:
            autocorr = 0.0
        # Vol ratio
        half = self._micro_window // 2
        vol_first = rets[:half].std()
        vol_second = rets[half:].std()
        vol_ratio = vol_second / vol_first if vol_first > 1e-15 else 1.0

        feats = np.array([[spread_mean, spread_cv, trade_rate, ofi_momentum,
                           autocorr, vol_ratio]])
        norm = self.micro_hmm.normalize(feats)
        posterior = self.micro_hmm.predict_proba_incremental(norm[0])
        self._micro_state = int(np.argmax(posterior))
        logger.info("Micro regime: state=%d (conf=%.2f)", self._micro_state, posterior.max())

    @property
    def macro_state(self) -> int:
        return self._macro_state

    @property
    def micro_state(self) -> int:
        return self._micro_state

    @property
    def is_warmed_up(self) -> bool:
        """Need at least one macro window (300 bars = 5 min) for regime to be valid."""
        return self._bars_seen >= self._macro_window

    def reset(self) -> None:
        """Reset for new session."""
        self._log_returns.clear()
        self._volumes.clear()
        self._spreads.clear()
        self._ofis.clear()
        self._bars_seen = 0
        self._macro_state = -1
        self._micro_state = -1
        self.macro_hmm.reset_filter()
        self.micro_hmm.reset_filter()


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: float
    exit_time: float
    side: Side
    entry_price: float
    exit_price: float
    pnl_ticks: float
    hold_seconds: float
    exit_reason: str
    entry_z: float


@dataclass
class RegimeModelConfig:
    """Configuration for one regime-specific model."""
    model_path: str
    regime_name: str  # e.g., "pair_3_0"
    macro_state: int  # which macro HMM state this model is for
    micro_state: int  # which micro HMM state
    entry_z: float
    sl_ticks: float
    trail_activation: float
    trail_distance: float


@dataclass
class BotConfig:
    """Full bot configuration."""
    # Models to run (regime-gated)
    models: list[RegimeModelConfig] = field(default_factory=list)

    # Shared params
    window_size: int = 30
    zscore_lookback: int = 300
    max_hold_seconds: int = 300
    min_hold_seconds: int = 30  # cooldown between trades
    daily_loss_limit: float = 200.0  # dollars
    max_consecutive_losses: int = 5
    cooldown_after_consec_losses: int = 300  # 5 min

    # Contract
    symbol: str = "MESM6"  # current front month

    @classmethod
    def default_paper(cls) -> BotConfig:
        """Default config with the 5 profitable regime models."""
        return cls(models=[
            RegimeModelConfig(
                model_path="models/regime_v2_fold2/w30_h5/pair_4_1.pt",
                regime_name="pair_4_1", macro_state=4, micro_state=1,
                entry_z=2.5, sl_ticks=8, trail_activation=12, trail_distance=6,
            ),
            RegimeModelConfig(
                model_path="models/regime_v2_fold2/w30_h5/pair_1_0.pt",
                regime_name="pair_1_0", macro_state=1, micro_state=0,
                entry_z=3.0, sl_ticks=8, trail_activation=12, trail_distance=2,
            ),
            RegimeModelConfig(
                model_path="models/regime_v2_fold2/w30_h5/pair_3_0.pt",
                regime_name="pair_3_0", macro_state=3, micro_state=0,
                entry_z=2.5, sl_ticks=12, trail_activation=9, trail_distance=2,
            ),
            RegimeModelConfig(
                model_path="models/regime_v2_fold2/w30_h5/pair_4_0.pt",
                regime_name="pair_4_0", macro_state=4, micro_state=0,
                entry_z=2.5, sl_ticks=12, trail_activation=12, trail_distance=4,
            ),
            RegimeModelConfig(
                model_path="models/regime_v2_fold2/w30_h5/pair_1_1.pt",
                regime_name="pair_1_1", macro_state=1, micro_state=1,
                entry_z=2.5, sl_ticks=9, trail_activation=12, trail_distance=6,
            ),
        ])


class LiveBot:
    """Main paper trading bot."""

    def __init__(
        self,
        config: BotConfig,
        executor: TradovateExecutor,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Streaming features
        self.features = StreamingFeatures(window_size=config.window_size)

        # Streaming regime detector
        self.regime = StreamingRegimeDetector()

        # Load all models
        self.models: dict[str, EntryModel] = {}
        for mc in config.models:
            model = EntryModel(n_features=15, seq_len=config.window_size).to(self.device)
            model.load_state_dict(torch.load(mc.model_path, weights_only=True, map_location=self.device))
            model.eval()
            self.models[mc.regime_name] = model
            logger.info("Loaded model %s from %s", mc.regime_name, mc.model_path)

        # Model configs by name
        self._model_configs = {mc.regime_name: mc for mc in config.models}

        # P(up) z-score buffers (one per model)
        self._p_up_buffers: dict[str, list[float]] = {mc.regime_name: [] for mc in config.models}

        # Trade state
        self._side = Side.FLAT
        self._entry_price: float = 0.0
        self._entry_time: float = 0.0
        self._entry_z: float = 0.0
        self._best_price: float = 0.0
        self._trail_active: bool = False
        self._active_sl_order_id: int | None = None
        self._active_model: str = ""
        self._contract_id: int | None = None
        self._cooldown_until: float = 0.0

        # P&L tracking
        self.trades: list[TradeRecord] = []
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._halted: bool = False

        # Stats
        self._ticks_received: int = 0
        self._bars_processed: int = 0
        self._signals_generated: int = 0
        self._running: bool = False

    async def run(self) -> None:
        """Start the bot. Connects to Databento and trades."""
        self._running = True
        logger.info("=" * 60)
        logger.info("LIVE BOT STARTING")
        logger.info("  Models: %s", [mc.regime_name for mc in self.config.models])
        logger.info("  Device: %s", self.device)
        logger.info("  Symbol: %s", self.config.symbol)
        logger.info("  Daily loss limit: $%.2f", self.config.daily_loss_limit)
        logger.info("=" * 60)

        # Authenticate with Tradovate
        self.executor.authenticate()
        self._contract_id = self.executor.find_contract(self.config.symbol)
        if self._contract_id is None:
            logger.error("Could not find contract %s", self.config.symbol)
            return

        # Connect to Databento live
        settings = DatabentoSettings()
        client = db.Live(settings.api_key)

        client.subscribe(
            dataset=settings.dataset,
            schema="tbbo",
            symbols=["MES.c.0"],
            stype_in="continuous",
        )

        client.add_callback(self._on_tick)
        client.start()

        logger.info("Live feed connected. Waiting for ticks...")

        # Run until stopped (signal handlers only on Unix)
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))
        except NotImplementedError:
            pass  # Windows — use Ctrl+C which raises KeyboardInterrupt

        try:
            while self._running:
                await asyncio.sleep(10)
                self._log_heartbeat()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            client.stop()
            self._close_position("shutdown")
            self._print_summary()

    async def _shutdown(self) -> None:
        logger.info("Shutdown requested")
        self._running = False

    def _on_tick(self, record: db.DBNRecord) -> None:
        """Callback for each live tick from Databento."""
        if not hasattr(record, "levels") or not record.levels:
            return

        bbo = record.levels[0]
        is_trade = isinstance(record, db.TBBOMsg)

        # Databento sends prices as fixed-point integers (1e-9 units)
        # Convert to actual dollar prices
        PRICE_SCALE = 1e-9

        tick = {
            "timestamp": record.ts_event,
            "bid_price": float(bbo.bid_px) * PRICE_SCALE,
            "bid_size": int(bbo.bid_sz),
            "ask_price": float(bbo.ask_px) * PRICE_SCALE,
            "ask_size": int(bbo.ask_sz),
            "trade_price": float(record.price) * PRICE_SCALE if is_trade else 0.0,
            "trade_size": int(record.size) if is_trade else 0,
            "trade_side": (1 if getattr(record, "side", "") == "A"
                          else -1 if getattr(record, "side", "") == "B" else 0),
        }

        self._ticks_received += 1

        if self._ticks_received == 1:
            logger.info("First tick received! bid=%.2f ask=%.2f",
                        tick["bid_price"], tick["ask_price"])
        if self._ticks_received <= 5 or self._ticks_received % 1000 == 0:
            logger.info("Tick %d: bid=%.2f ask=%.2f", self._ticks_received,
                        tick["bid_price"], tick["ask_price"])

        # Feed to streaming features
        bar_completed = self.features.on_tick(tick)

        # Check for new bar (by comparing bars processed)
        if self.features.n_bars > self._bars_processed:
            self._bars_processed = self.features.n_bars
            self._on_new_bar()

    def _on_new_bar(self) -> None:
        """Called every second when a new 1-sec bar completes."""
        window = self.features.get_window()
        if window is None:
            return

        now = time.time()

        # Feed regime detector (uses bar-level data from streaming features)
        sf = self.features
        if sf._prev_mid > 0:
            log_ret = sf._ret_history[-1] if sf._ret_history else 0.0
            volume = sf._tc_history[-1] if sf._tc_history else 0.0
            spread = (sf.latest_ask - sf.latest_bid) / MES_TICK
            ofi = sf._ofi_history[-1] if sf._ofi_history else 0.0
            self.regime.on_bar(log_ret, volume, spread, ofi)

        # Check circuit breakers
        if self._halted:
            return

        if self._daily_pnl <= -self.config.daily_loss_limit:
            if not self._halted:
                logger.warning("DAILY LOSS LIMIT HIT: $%.2f. Halting.", self._daily_pnl)
                self._halted = True
                self._close_position("daily_limit")
            return

        # Run inference on all models
        with torch.no_grad():
            input_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)

            for name, model in self.models.items():
                mc = self._model_configs[name]

                # REGIME GATE: only process models whose regime is active
                if self.regime.is_warmed_up:
                    if self.regime.macro_state != mc.macro_state or self.regime.micro_state != mc.micro_state:
                        # Not in this model's regime — skip inference
                        # But still check exit if we have a position from this model
                        if self._side != Side.FLAT and name == self._active_model:
                            self._check_exit(now)
                        continue

                dir_logits, _ = model(input_tensor)
                p_up = float(torch.sigmoid(dir_logits.squeeze()).cpu())

                # Update z-score buffer
                buf = self._p_up_buffers[name]
                buf.append(p_up)
                if len(buf) > self.config.zscore_lookback:
                    buf[:] = buf[-self.config.zscore_lookback:]

                # Need enough samples for stable z-score
                min_zscore_samples = min(self.config.zscore_lookback, 120)
                if len(buf) < min_zscore_samples:
                    continue

                # Compute z-score over lookback window
                arr = np.array(buf[-self.config.zscore_lookback:])
                mean = arr.mean()
                std = arr.std()
                if std < 1e-10:
                    continue
                z = (p_up - mean) / std

                self._signals_generated += 1

                # If we have a position, check trailing stop
                if self._side != Side.FLAT and name == self._active_model:
                    self._check_exit(now)

                # If flat, check entry (only if not in cooldown)
                if self._side == Side.FLAT and now >= self._cooldown_until:
                    if z >= mc.entry_z:
                        self._enter(Side.LONG, mc, z, now)
                    elif z <= -mc.entry_z:
                        self._enter(Side.SHORT, mc, z, now)

    def _enter(self, side: Side, mc: RegimeModelConfig, z: float, now: float) -> None:
        """Enter a new position."""
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            if now < self._cooldown_until:
                return
            logger.info("Consecutive loss cooldown expired, allowing trades again")
            self._consecutive_losses = 0

        action = "Buy" if side == Side.LONG else "Sell"

        try:
            result = self.executor.place_market_order(self._contract_id, action)
            order_id = result.get("orderId")
            logger.info("ENTRY: %s %s (z=%.2f, model=%s, order=%s)",
                        action, mc.regime_name, z, mc.regime_name, order_id)
        except Exception as e:
            logger.error("Failed to place entry order: %s", e)
            return

        self._side = side
        self._entry_price = self.features.latest_ask if side == Side.LONG else self.features.latest_bid
        self._entry_time = now
        self._entry_z = z
        self._best_price = self.features.latest_mid
        self._trail_active = False
        self._active_model = mc.regime_name

        # No exchange-side SL — managed internally to avoid double-exit issues
        logger.info("  Internal SL: %.1f ticks, trail: act=%.1f dist=%.1f",
                     mc.sl_ticks, mc.trail_activation, mc.trail_distance)

    def _check_exit(self, now: float) -> None:
        """Check hard SL, trailing stop, and max hold for active position."""
        mc = self._model_configs[self._active_model]
        mid = self.features.latest_mid
        hold = now - self._entry_time

        # Update best price
        if self._side == Side.LONG:
            self._best_price = max(self._best_price, mid)
            runup = (self._best_price - self._entry_price) / MES_TICK
            current_pnl = (mid - self._entry_price) / MES_TICK
        else:
            self._best_price = min(self._best_price, mid)
            runup = (self._entry_price - self._best_price) / MES_TICK
            current_pnl = (self._entry_price - mid) / MES_TICK

        # Hard SL (internal check — backup for exchange SL, required for dry run)
        if current_pnl <= -mc.sl_ticks:
            logger.info("  Hard SL hit (pnl=%.1f ticks, SL=%.1f)", current_pnl, -mc.sl_ticks)
            self._close_position("hard_sl")
            return

        # Activate trail
        if not self._trail_active and runup >= mc.trail_activation:
            self._trail_active = True
            logger.info("  Trail activated (runup=%.1f ticks)", runup)

        # Check trail
        if self._trail_active:
            drawback = runup - current_pnl
            if drawback >= mc.trail_distance:
                self._close_position("trail")
                return

        # Max hold
        if hold >= self.config.max_hold_seconds:
            self._close_position("max_hold")

    def _close_position(self, reason: str) -> None:
        """Close the current position."""
        if self._side == Side.FLAT:
            return

        now = time.time()
        mid = self.features.latest_mid

        # Close via market order
        action = "Sell" if self._side == Side.LONG else "Buy"
        try:
            close_result = self.executor.place_market_order(self._contract_id, action)
            logger.info("  Close order: %s", close_result.get("orderId", "?"))
        except Exception as e:
            logger.error("Failed to close position: %s", e)
            # Try flatten as fallback
            try:
                self.executor.flatten_position(self._contract_id)
            except Exception:
                logger.error("CRITICAL: Could not flatten position!")

        # Calculate P&L using bid/ask (best estimate without fill confirmation)
        if self._side == Side.LONG:
            exit_price = self.features.latest_bid
            pnl_ticks = (exit_price - self._entry_price) / MES_TICK
        else:
            exit_price = self.features.latest_ask
            pnl_ticks = (self._entry_price - exit_price) / MES_TICK

        pnl_dollars = pnl_ticks * 1.25  # MES tick value
        self._daily_pnl += pnl_dollars

        trade = TradeRecord(
            entry_time=self._entry_time, exit_time=now,
            side=self._side, entry_price=self._entry_price,
            exit_price=mid, pnl_ticks=pnl_ticks,
            hold_seconds=now - self._entry_time,
            exit_reason=reason, entry_z=self._entry_z,
        )
        self.trades.append(trade)

        # Consecutive loss tracking
        if pnl_ticks <= 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._cooldown_until = now + self.config.cooldown_after_consec_losses
                logger.warning("  %d consecutive losses — cooldown until %ds",
                               self._consecutive_losses, self.config.cooldown_after_consec_losses)
        else:
            self._consecutive_losses = 0

        logger.info("EXIT [%s]: %s %.1f ticks ($%.2f) | hold=%.0fs | model=%s | daily=$%.2f",
                     reason, self._side.name, pnl_ticks, pnl_dollars,
                     now - self._entry_time, self._active_model, self._daily_pnl)

        self._side = Side.FLAT
        self._cooldown_until = max(self._cooldown_until, now + self.config.min_hold_seconds)

    def _log_heartbeat(self) -> None:
        """Periodic heartbeat log every 10 seconds."""
        mid = self.features.latest_mid
        bid = self.features.latest_bid
        ask = self.features.latest_ask

        # Position info
        if self._side != Side.FLAT:
            if self._side == Side.LONG:
                unrealized = (mid - self._entry_price) / MES_TICK
            else:
                unrealized = (self._entry_price - mid) / MES_TICK
            hold = time.time() - self._entry_time
            pos_str = f"{self._side.name} unrealized={unrealized:+.1f}t hold={hold:.0f}s trail={'ON' if self._trail_active else 'off'}"
        else:
            pos_str = "FLAT"

        # Regime info
        if self.regime.is_warmed_up:
            regime_str = f"macro={self.regime.macro_state} micro={self.regime.micro_state}"
        else:
            regime_str = f"warming up ({self.regime._bars_seen}/300 bars)"

        # Pull real P&L from Tradovate every ~60s
        tv_pnl_str = ""
        if self._bars_processed % 60 < 10:  # roughly every 60 heartbeats
            try:
                acct = self.executor.get_account_pnl()
                if acct:
                    tv_pnl_str = f" | TV: realized=${acct.get('realized_pnl', '?')} open=${acct.get('open_pnl', '?')}"
            except Exception:
                pass

        logger.info(
            "HEARTBEAT | mid=%.2f bid=%.2f ask=%.2f | %s | regime=[%s] | "
            "ticks=%d bars=%d signals=%d trades=%d daily=$%.2f%s",
            mid, bid, ask, pos_str, regime_str,
            self._ticks_received, self._bars_processed,
            self._signals_generated, len(self.trades), self._daily_pnl,
            tv_pnl_str,
        )

    def _print_summary(self) -> None:
        """Print session summary."""
        n = len(self.trades)
        if n == 0:
            logger.info("Session ended. No trades.")
            return

        pnls = [t.pnl_ticks for t in self.trades]
        winners = [p for p in pnls if p > 0]

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("  Trades: %d", n)
        logger.info("  Win rate: %.1f%%", len(winners) / n * 100 if n > 0 else 0)
        logger.info("  Total P&L: %.1f ticks ($%.2f)", sum(pnls), sum(pnls) * 1.25)
        logger.info("  Daily P&L: $%.2f", self._daily_pnl)
        for t in self.trades:
            logger.info("    %s %s: %.1ft ($%.2f) hold=%.0fs exit=%s model=%s",
                        t.side.name, "WIN" if t.pnl_ticks > 0 else "LOSS",
                        t.pnl_ticks, t.pnl_ticks * 1.25, t.hold_seconds,
                        t.exit_reason, "")
        logger.info("=" * 60)
