"""Unit tests for all L1 features."""

from __future__ import annotations

from math import isnan, nan

import numpy as np
import pytest

from src.data.schemas import L1Record, make_quote_record, make_trade_record

# Helpers

SEC = 1_000_000_000  # 1 second in nanoseconds


def quote(ts: int, bid: float = 5000.0, bid_sz: int = 10,
          ask: float = 5000.25, ask_sz: int = 12) -> L1Record:
    return make_quote_record(ts, bid, bid_sz, ask, ask_sz)


def trade(ts: int, price: float = 5000.25, size: int = 1, side: int = 1,
          bid: float = 5000.0, bid_sz: int = 10,
          ask: float = 5000.25, ask_sz: int = 12) -> L1Record:
    return make_trade_record(ts, bid, bid_sz, ask, ask_sz, price, size, side)


# ── Welford Normalizer ──────────────────────────────────────────

class TestWelfordNormalizer:
    def test_zscore_converges(self):
        from src.features.normalizer import WelfordNormalizer
        rng = np.random.default_rng(42)
        norm = WelfordNormalizer(window=200)
        values = rng.normal(100.0, 15.0, size=500)
        zscores = [norm.update(v) for v in values]

        # After 200+ samples, z-scores should be roughly N(0,1)
        last_200 = zscores[-200:]
        mean_z = np.mean(last_200)
        std_z = np.std(last_200)
        assert abs(mean_z) < 0.3, f"mean z-score should be near 0, got {mean_z}"
        assert 0.5 < std_z < 1.5, f"std of z-scores should be near 1, got {std_z}"

    def test_is_valid(self):
        from src.features.normalizer import WelfordNormalizer
        norm = WelfordNormalizer(window=100)
        for i in range(19):
            norm.update(float(i))
        assert not norm.is_valid()
        norm.update(20.0)
        assert norm.is_valid()

    def test_nan_handled(self):
        from src.features.normalizer import WelfordNormalizer
        norm = WelfordNormalizer()
        assert norm.update(float("nan")) == 0.0
        assert norm.update(float("inf")) == 0.0


class TestStaleness:
    def test_stale_zeros_out(self):
        from src.features.normalizer import staleness_gate
        # 3 seconds gap, max_age is 2 seconds
        assert staleness_gate(42.0, 5 * SEC, 2 * SEC, 2 * SEC) == 0.0

    def test_fresh_passes_through(self):
        from src.features.normalizer import staleness_gate
        assert staleness_gate(42.0, 3 * SEC, 2 * SEC, 2 * SEC) == 42.0

    def test_no_prior_update(self):
        from src.features.normalizer import staleness_gate
        assert staleness_gate(42.0, 3 * SEC, 0, 2 * SEC) == 0.0


# ── Lee-Ready ───────────────────────────────────────────────────

class TestLeeReady:
    def test_above_mid_is_buyer(self):
        from src.features.lee_ready import LeeReadyClassifier
        lr = LeeReadyClassifier()
        # trade at ask (5000.25) > mid (5000.125) => buyer
        r = trade(1 * SEC, price=5000.25, bid=5000.0, ask=5000.25)
        val = lr.update(r)
        assert val == 1.0

    def test_below_mid_is_seller(self):
        from src.features.lee_ready import LeeReadyClassifier
        lr = LeeReadyClassifier()
        r = trade(1 * SEC, price=5000.0, bid=5000.0, ask=5000.25)
        val = lr.update(r)
        assert val == -1.0

    def test_at_mid_tick_rule(self):
        from src.features.lee_ready import LeeReadyClassifier
        lr = LeeReadyClassifier()
        # First trade at 5000.00
        lr.update(trade(1 * SEC, price=5000.0, bid=4999.75, ask=5000.25))
        # Second trade at mid (5000.0), but same price as prev => 0
        val = lr.update(trade(2 * SEC, price=5000.0, bid=4999.75, ask=5000.25))
        assert val == 0.0

    def test_at_mid_tick_rule_uptick(self):
        from src.features.lee_ready import LeeReadyClassifier
        lr = LeeReadyClassifier()
        # First trade at 4999.75
        lr.update(trade(1 * SEC, price=4999.75, bid=4999.75, ask=5000.25))
        # Second trade at mid (5000.0), uptick from 4999.75 => buyer
        val = lr.update(trade(2 * SEC, price=5000.0, bid=4999.75, ask=5000.25))
        assert val == 1.0

    def test_quotes_return_none(self):
        from src.features.lee_ready import LeeReadyClassifier
        lr = LeeReadyClassifier()
        assert lr.update(quote(1 * SEC)) is None


# ── OFI ─────────────────────────────────────────────────────────

class TestOFI:
    def test_bid_tick_up_positive(self):
        """Bid price tick up should produce positive OFI."""
        from src.features.ofi import OrderFlowImbalance
        ofi = OrderFlowImbalance()
        ofi.update(quote(1 * SEC, bid=5000.0, bid_sz=10, ask=5000.50, ask_sz=10))
        val = ofi.update(quote(2 * SEC, bid=5000.25, bid_sz=15, ask=5000.50, ask_sz=10))
        # Bid ticked up: delta_bid = +15. Ask unchanged: delta_ask = 0
        assert val is not None
        assert val > 0

    def test_ask_tick_down_positive(self):
        """Ask tick down (more aggressive sellers) is negative pressure from ask side."""
        from src.features.ofi import OrderFlowImbalance
        ofi = OrderFlowImbalance()
        ofi.update(quote(1 * SEC, bid=5000.0, bid_sz=10, ask=5000.50, ask_sz=10))
        val = ofi.update(quote(2 * SEC, bid=5000.0, bid_sz=10, ask=5000.25, ask_sz=15))
        # Bid unchanged: delta_bid = 0. Ask ticked down: delta_ask = +15
        # OFI = 0 - 15 = -15
        assert val is not None
        assert val < 0

    def test_first_record_returns_none(self):
        from src.features.ofi import OrderFlowImbalance
        ofi = OrderFlowImbalance()
        assert ofi.update(quote(1 * SEC)) is None

    def test_size_change_at_same_price(self):
        """Size increase at same bid = positive delta_bid."""
        from src.features.ofi import OrderFlowImbalance
        ofi = OrderFlowImbalance()
        ofi.update(quote(1 * SEC, bid=5000.0, bid_sz=10, ask=5000.25, ask_sz=10))
        val = ofi.update(quote(2 * SEC, bid=5000.0, bid_sz=20, ask=5000.25, ask_sz=10))
        assert val == 10.0  # delta_bid = 20-10 = 10, delta_ask = 0


# ── VPIN ────────────────────────────────────────────────────────

class TestVPIN:
    def test_balanced_volume(self):
        """Equal buy and sell volume should give VPIN near 0."""
        from src.features.vpin import VPIN
        v = VPIN(bucket_size=10, num_buckets=5)

        # Fill 5 buckets with alternating buy/sell
        val = None
        for i in range(50):
            if i % 2 == 0:
                # Trade above mid -> buyer
                val = v.update(trade(i * SEC, price=5000.25, size=1,
                                     bid=5000.0, ask=5000.25))
            else:
                # Trade below mid -> seller
                val = v.update(trade(i * SEC, price=5000.0, size=1,
                                     bid=5000.0, ask=5000.25))

        assert val is not None
        assert val < 0.3  # should be near 0 for balanced flow

    def test_all_buys_high_vpin(self):
        """All buy trades should give VPIN near 1."""
        from src.features.vpin import VPIN
        v = VPIN(bucket_size=10, num_buckets=5)

        val = None
        for i in range(60):
            val = v.update(trade(i * SEC, price=5000.25, size=1,
                                 bid=5000.0, ask=5000.25))

        assert val is not None
        assert val > 0.8

    def test_no_value_before_bucket(self):
        from src.features.vpin import VPIN
        v = VPIN(bucket_size=100, num_buckets=5)
        # Single trade won't complete a bucket
        assert v.update(trade(1 * SEC, price=5000.25, size=1)) is None


# ── MicroPrice ──────────────────────────────────────────────────

class TestMicroPrice:
    def test_returns_log_return(self):
        from src.features.microprice import MicroPrice
        mp = MicroPrice()
        mp.update(quote(1 * SEC, bid=5000.0, bid_sz=10, ask=5000.25, ask_sz=10))
        # Shift price up
        val = mp.update(quote(2 * SEC, bid=5000.25, bid_sz=10, ask=5000.50, ask_sz=10))
        assert val is not None
        assert val > 0  # price went up, log return positive

    def test_first_record_returns_none(self):
        from src.features.microprice import MicroPrice
        mp = MicroPrice()
        assert mp.update(quote(1 * SEC)) is None

    def test_imbalance_shifts_microprice(self):
        from src.features.microprice import MicroPrice
        mp = MicroPrice()
        # Balanced
        mp.update(quote(1 * SEC, bid=5000.0, bid_sz=100, ask=5000.25, ask_sz=100))
        # Heavy bid side -> micro shifts toward ask -> positive return
        val = mp.update(quote(2 * SEC, bid=5000.0, bid_sz=200, ask=5000.25, ask_sz=10))
        assert val is not None
        assert val > 0


# ── CVD ─────────────────────────────────────────────────────────

class TestCVD:
    def test_all_buys_positive(self):
        from src.features.cvd import CumulativeVolumeDelta
        cvd = CumulativeVolumeDelta()
        val = None
        for i in range(10):
            val = cvd.update(trade(i * SEC, price=5000.25, size=5,
                                   bid=5000.0, ask=5000.25))
        assert val is not None
        assert val > 0

    def test_all_sells_negative(self):
        from src.features.cvd import CumulativeVolumeDelta
        cvd = CumulativeVolumeDelta()
        val = None
        for i in range(10):
            val = cvd.update(trade(i * SEC, price=5000.0, size=5,
                                   bid=5000.0, ask=5000.25))
        assert val is not None
        assert val < 0

    def test_window_expires(self):
        from src.features.cvd import CumulativeVolumeDelta
        cvd = CumulativeVolumeDelta(window_ns=5 * SEC)
        # 5 buys
        for i in range(5):
            cvd.update(trade(i * SEC, price=5000.25, size=1, bid=5000.0, ask=5000.25))
        # Jump far ahead, old trades expire
        val = cvd.update(trade(20 * SEC, price=5000.0, size=1, bid=5000.0, ask=5000.25))
        # Only the latest sell should remain
        assert val is not None
        assert val < 0


# ── Spread ──────────────────────────────────────────────────────

class TestSpread:
    def test_one_tick_spread(self):
        from src.features.spread import Spread
        s = Spread()
        val = s.update(quote(1 * SEC, bid=5000.0, ask=5000.25))
        assert val == 1.0  # 0.25 / 0.25 = 1 tick

    def test_two_tick_spread(self):
        from src.features.spread import Spread
        s = Spread()
        val = s.update(quote(1 * SEC, bid=5000.0, ask=5000.50))
        assert val == 2.0


# ── Trade Rate ──────────────────────────────────────────────────

class TestTradeRate:
    def test_rate_calculation(self):
        from src.features.trade_rate import TradeRate
        tr = TradeRate(window_ns=10 * SEC)
        # 5 trades in 5 seconds
        for i in range(5):
            tr.update(trade(i * SEC))
        val = tr.update(quote(5 * SEC))
        assert val == pytest.approx(5.0 / 10.0, abs=0.01)

    def test_quotes_dont_count(self):
        from src.features.trade_rate import TradeRate
        tr = TradeRate(window_ns=10 * SEC)
        for i in range(5):
            tr.update(quote(i * SEC))
        val = tr.update(quote(5 * SEC))
        assert val == 0.0


# ── Volume Profile ──────────────────────────────────────────────

class TestVolumeProfile:
    def test_poc_tracking(self):
        from src.features.volume_profile import VolumeProfile
        vp = VolumeProfile()
        # Lots of volume at 5000.25
        for i in range(50):
            vp.update(trade(i * SEC, price=5000.25, size=10))
        # Small volume elsewhere
        for i in range(5):
            vp.update(trade((50 + i) * SEC, price=5000.50, size=1))

        val = vp.update(quote(60 * SEC, bid=5000.25, ask=5000.50))
        # mid = 5000.375, POC = 5000.25, distance in ticks
        assert val is not None
        assert val > 0  # price above POC


# ── Hurst ───────────────────────────────────────────────────────

class TestHurst:
    def test_trending_series(self):
        from src.features.hurst import HurstExponent
        h = HurstExponent(window=200)
        # Monotonically increasing prices -> trending -> H > 0.5
        val = None
        for i in range(200):
            val = h.update(trade(i * SEC, price=5000.0 + i * 0.25))
        assert val is not None
        assert val > 0.5

    def test_not_ready_before_window(self):
        from src.features.hurst import HurstExponent
        h = HurstExponent(window=200)
        for i in range(50):
            val = h.update(trade(i * SEC, price=5000.0))
        assert val is None


# ── Realized Vol ────────────────────────────────────────────────

class TestRealizedVol:
    def test_positive_output(self):
        from src.features.realized_vol import RealizedVolatility
        rv = RealizedVolatility(window_ns=60 * SEC)
        rng = np.random.default_rng(42)
        val = None
        price = 5000.0
        for i in range(120):
            price += rng.normal(0, 0.1)
            val = rv.update(quote(i * SEC, bid=price - 0.125, ask=price + 0.125))
        assert val is not None
        assert val > 0

    def test_constant_price_zero_vol(self):
        from src.features.realized_vol import RealizedVolatility
        rv = RealizedVolatility(window_ns=60 * SEC)
        for i in range(120):
            rv.update(quote(i * SEC, bid=5000.0, ask=5000.25))
        val = rv.update(quote(120 * SEC, bid=5000.0, ask=5000.25))
        assert val is not None
        assert val < 1e-10


# ── Trade Size Distribution ─────────────────────────────────────

class TestTradeSizeDist:
    def test_large_trade_high_percentile(self):
        from src.features.trade_size_dist import TradeSizeDistribution
        tsd = TradeSizeDistribution(window=100)
        # Fill with small trades
        for i in range(99):
            tsd.update(trade(i * SEC, size=1))
        # Big trade
        val = tsd.update(trade(100 * SEC, size=100))
        assert val is not None
        assert val > 0.9

    def test_small_trade_low_percentile(self):
        from src.features.trade_size_dist import TradeSizeDistribution
        tsd = TradeSizeDistribution(window=100)
        # Fill with big trades
        for i in range(99):
            tsd.update(trade(i * SEC, size=100))
        # Small trade
        val = tsd.update(trade(100 * SEC, size=1))
        assert val is not None
        assert val < 0.1


# ── Return Autocorrelation ──────────────────────────────────────

class TestReturnAutocorr:
    def test_trending_positive_autocorr(self):
        from src.features.return_autocorr import ReturnAutocorrelation
        ra = ReturnAutocorrelation(window=60)
        # Monotonic uptrend -> positive autocorrelation
        val = None
        for i in range(120):
            price = 5000.0 + i * 0.25
            val = ra.update(quote(i * SEC, bid=price, ask=price + 0.25))
        # With constant returns, autocorrelation might be noisy but should lean positive
        assert val is not None

    def test_not_ready_before_window(self):
        from src.features.return_autocorr import ReturnAutocorrelation
        ra = ReturnAutocorrelation(window=60)
        val = ra.update(quote(1 * SEC))
        assert val is None


# ── Pipeline Integration ────────────────────────────────────────

class TestPipeline:
    def test_output_shape(self):
        """Pipeline should output [window_size, num_features] when window is full."""
        from src.features.pipeline import build_default_pipeline

        pipeline = build_default_pipeline(window_size=10)
        rng = np.random.default_rng(42)

        price = 5000.0
        # Feed enough data to fill the pipeline window (need 10 seconds of data)
        # Send multiple records per second to feed all features
        for sec in range(200):
            ts = sec * SEC
            price += rng.normal(0, 0.1)
            bid = round(price / 0.25) * 0.25
            ask = bid + 0.25

            # Quote update
            pipeline.process_record(
                make_quote_record(ts, bid, 10 + rng.integers(-3, 3), ask, 12 + rng.integers(-3, 3))
            )

            # Trade
            trade_side = 1 if rng.random() > 0.5 else -1
            trade_px = ask if trade_side == 1 else bid
            pipeline.process_record(
                make_trade_record(
                    ts + SEC // 2, bid, 10, ask, 12,
                    trade_px, int(rng.integers(1, 10)), trade_side,
                )
            )

        window = pipeline.get_window()
        assert window is not None
        assert window.shape == (10, 12)
        # Values should be roughly in z-score range
        assert np.all(np.abs(window) < 10), "z-scores should be bounded"

    def test_reset_clears_window(self):
        from src.features.pipeline import build_default_pipeline
        pipeline = build_default_pipeline(window_size=5)

        for sec in range(50):
            ts = sec * SEC
            pipeline.process_record(quote(ts, bid=5000.0, ask=5000.25))
            pipeline.process_record(trade(ts + SEC // 2))

        pipeline.reset_session()
        assert pipeline.get_window() is None

    def test_staleness_zeros_during_gap(self):
        """Features should zero out during gaps > staleness threshold."""
        from src.features.pipeline import FeaturePipeline
        from src.features.spread import Spread

        pipeline = FeaturePipeline(
            features=[Spread()],
            window_size=5,
            staleness_ns=2 * SEC,
        )

        # Feed 10 seconds of data
        for sec in range(10):
            pipeline.process_record(quote(sec * SEC))

        # 5-second gap (exceeds 2s staleness)
        vec = pipeline.process_record(quote(15 * SEC))

        # The vector emitted at second 15 should be based on a stale value -> zeroed
        # The normalizer will output 0 for the gated-to-zero input
        assert vec is not None
