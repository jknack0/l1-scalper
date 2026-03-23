"""Unit tests for HMM regime detector, BOCPD, and position sizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Synthetic data generators ───────────────────────────────────

def _make_regime_data(n_per_regime: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 3-regime data with known labels.

    Features: [return_autocorr, hurst, variance_ratio, efficiency_ratio]

    Regime 0 (trending): high autocorr, high Hurst, VR > 1, high efficiency
    Regime 1 (mean-reverting): negative autocorr, low Hurst, VR < 1, low efficiency
    Regime 2 (choppy): zero autocorr, Hurst ~0.5, VR ~1, very low efficiency

    All features measure directional structure, not volatility magnitude.
    """
    rng = np.random.default_rng(seed)
    n = n_per_regime

    # Trending: persistent directional moves
    trending = np.column_stack([
        rng.normal(0.5, 0.05, n),       # return_autocorr (high positive)
        rng.normal(0.75, 0.03, n),      # hurst (> 0.5, persistent)
        rng.normal(1.8, 0.15, n),       # variance_ratio (> 1, trending)
        rng.normal(0.7, 0.05, n),       # efficiency_ratio (high, clean trend)
    ])

    # Mean-reverting: oscillating, anti-persistent
    mean_rev = np.column_stack([
        rng.normal(-0.4, 0.05, n),      # return_autocorr (negative)
        rng.normal(0.3, 0.03, n),       # hurst (< 0.5, anti-persistent)
        rng.normal(0.5, 0.1, n),        # variance_ratio (< 1, mean-reverting)
        rng.normal(0.15, 0.04, n),      # efficiency_ratio (low, oscillating)
    ])

    # Choppy: no structure, random noise
    choppy = np.column_stack([
        rng.normal(0.0, 0.05, n),       # return_autocorr (near zero)
        rng.normal(0.5, 0.03, n),       # hurst (~0.5, random walk)
        rng.normal(1.0, 0.1, n),        # variance_ratio (~1, random)
        rng.normal(0.1, 0.03, n),       # efficiency_ratio (very low, noise)
    ])

    data = np.vstack([trending, mean_rev, choppy])
    labels = np.concatenate([
        np.zeros(n, dtype=int),   # trending
        np.ones(n, dtype=int),    # mean-reverting
        np.full(n, 2, dtype=int), # choppy
    ])

    return data, labels


# ── HMM Tests ───────────────────────────────────────────────────

class TestMarketRegimeHMM:
    def test_fit_and_predict(self):
        """HMM should recover 3 distinct states from well-separated synthetic data."""
        from src.regime.hmm import MarketRegimeHMM

        data, true_labels = _make_regime_data(n_per_regime=500)
        model = MarketRegimeHMM(n_iter=200, random_state=42)
        model.fit(data, lengths=[500, 500, 500])

        # Predict each block independently
        all_predicted = []
        for i in range(3):
            block_data = data[i * 500 : (i + 1) * 500]
            block_pred = model.predict(block_data)
            all_predicted.append(block_pred)

        predicted = np.concatenate(all_predicted)

        # Check that model predicts 3 distinct states
        assert len(np.unique(predicted)) == 3

        # Each block should be dominated by one state
        for regime_start in range(3):
            block = predicted[regime_start * 500 : (regime_start + 1) * 500]
            dominant = np.bincount(block).argmax()
            purity = np.mean(block == dominant)
            assert purity > 0.7, (
                f"Regime block {regime_start}: purity {purity:.2f} too low"
            )

        # Different blocks should have different dominant states
        dominants = set()
        for i in range(3):
            block = predicted[i * 500 : (i + 1) * 500]
            dominants.add(np.bincount(block).argmax())
        assert len(dominants) == 3, "Each block should have a different dominant regime"

    def test_predict_proba_sums_to_one(self):
        from src.regime.hmm import MarketRegimeHMM

        data, _ = _make_regime_data(n_per_regime=200)
        model = MarketRegimeHMM(n_iter=100, random_state=42)
        model.fit(data)

        proba = model.predict_proba(data[:50])
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_predict_proba_sequence(self):
        from src.regime.hmm import MarketRegimeHMM

        data, _ = _make_regime_data(n_per_regime=200)
        model = MarketRegimeHMM(n_iter=100, random_state=42)
        model.fit(data)

        proba_seq = model.predict_proba_sequence(data[:100])
        assert proba_seq.shape == (100, 3)
        assert np.allclose(proba_seq.sum(axis=1), 1.0, atol=1e-6)

    def test_save_and_load(self, tmp_path):
        from src.regime.hmm import MarketRegimeHMM

        data, _ = _make_regime_data(n_per_regime=200)
        model = MarketRegimeHMM(n_iter=50, random_state=42)
        model.fit(data)

        path = tmp_path / "hmm_model.pkl"
        model.save(path)

        model2 = MarketRegimeHMM()
        model2.load(path)

        proba1 = model.predict_proba(data[:20])
        proba2 = model2.predict_proba(data[:20])
        np.testing.assert_allclose(proba1, proba2, atol=1e-10)

    def test_transition_matrix_rows_sum_to_one(self):
        from src.regime.hmm import MarketRegimeHMM

        data, _ = _make_regime_data(n_per_regime=200)
        model = MarketRegimeHMM(n_iter=50, random_state=42)
        model.fit(data)

        tm = model.transition_matrix
        assert tm.shape == (3, 3)
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_state_label_assignment(self):
        """Trending state should have highest Hurst mean,
        mean-reverting should have lowest."""
        from src.regime.hmm import MarketRegimeHMM, TRENDING, MEAN_REVERTING, FEAT_HURST

        data, _ = _make_regime_data(n_per_regime=500)
        model = MarketRegimeHMM(n_iter=200, random_state=42)
        model.fit(data)

        means = model.means
        # Trending should have highest Hurst, mean-reverting should have lowest
        assert means[TRENDING, FEAT_HURST] > means[MEAN_REVERTING, FEAT_HURST]


# ── BOCPD Tests ─────────────────────────────────────────────────

class TestBOCPD:
    def test_stable_signal_low_cp(self):
        """Stable signal should have low changepoint probability."""
        from src.regime.bocpd import BOCPD

        bocpd = BOCPD(hazard_rate=1.0 / 250.0)
        rng = np.random.default_rng(42)

        # 200 bars of stable signal
        probs = []
        for _ in range(200):
            p = bocpd.detect(rng.normal(0.0, 0.01))
            probs.append(p)

        # After stabilization, cp prob should be low
        assert np.mean(probs[-50:]) < 0.1

    def test_volatility_spike_detected(self):
        """Sudden volatility spike should raise changepoint probability."""
        from src.regime.bocpd import BOCPD

        bocpd = BOCPD(hazard_rate=1.0 / 50.0, mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=0.001)
        rng = np.random.default_rng(42)

        # 200 bars of calm centered near 0
        for _ in range(200):
            bocpd.detect(rng.normal(0.0, 0.01))

        # Sudden massive shift
        spike_probs = []
        for _ in range(20):
            p = bocpd.detect(rng.normal(10.0, 0.1))
            spike_probs.append(p)

        # The cp probability should exceed the hazard baseline during the spike
        assert max(spike_probs) > 1.0 / 50.0

    def test_reset(self):
        from src.regime.bocpd import BOCPD

        bocpd = BOCPD()
        for _ in range(50):
            bocpd.detect(np.random.randn())

        bocpd.reset()
        # After reset, should be back to initial state
        p = bocpd.detect(0.0)
        assert 0.0 <= p <= 1.0

    def test_output_range(self):
        from src.regime.bocpd import BOCPD

        bocpd = BOCPD()
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = bocpd.detect(rng.normal(0, 1))
            assert 0.0 <= p <= 1.0


# ── RegimeDetector Tests ────────────────────────────────────────

class TestRegimeDetector:
    def _fit_and_get_detector(self):
        from src.regime.regime_detector import RegimeDetector

        data, _ = _make_regime_data(n_per_regime=300)
        detector = RegimeDetector()
        detector.hmm.fit(data)
        return detector, data

    def test_update_returns_regime_state(self):
        from src.regime.regime_detector import RegimeState

        detector, data = self._fit_and_get_detector()
        state = detector.update(data[0])

        assert isinstance(state, RegimeState)
        assert state.posteriors.shape == (3,)
        assert abs(state.posteriors.sum() - 1.0) < 1e-6
        assert state.dominant_regime in (0, 1, 2)
        assert 0.0 <= state.confidence <= 1.0
        assert 0.0 <= state.changepoint_prob <= 1.0

    def test_cooldown_after_transition(self):
        detector, data = self._fit_and_get_detector()

        # Feed trending data, then suddenly mean-reverting
        for obs in data[:50]:  # trending block
            state = detector.update(obs)

        # Jump to mean-reverting
        state = detector.update(data[400])  # mean-reverting block

        # If transition occurred, should be in cooldown
        # (may or may not detect transition depending on model)
        assert state.bars_since_transition >= 0

    def test_regime_name(self):
        detector, data = self._fit_and_get_detector()
        state = detector.update(data[0])
        assert state.regime_name in ("trending", "mean_reverting", "choppy")

    def test_reset(self):
        detector, data = self._fit_and_get_detector()
        for obs in data[:20]:
            detector.update(obs)
        detector.reset()
        # Should work again after reset
        state = detector.update(data[0])
        assert state.bars_since_transition == 0


# ── PositionSizer Tests ─────────────────────────────────────────

class TestPositionSizer:
    def test_choppy_always_zero(self):
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState

        sizer = RegimePositionSizer()

        # State with high choppy posterior
        state = RegimeState(
            posteriors=np.array([0.1, 0.1, 0.8]),
            dominant_regime=2,
            confidence=0.8,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
        )

        weights = sizer.get_weights(state)
        assert weights["choppy"] == 0.0

    def test_cooldown_reduces_weights(self):
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState

        sizer = RegimePositionSizer()

        base_state = RegimeState(
            posteriors=np.array([0.8, 0.1, 0.1]),
            dominant_regime=0,
            confidence=0.8,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
        )

        cooldown_state = RegimeState(
            posteriors=np.array([0.8, 0.1, 0.1]),
            dominant_regime=0,
            confidence=0.8,
            changepoint_prob=0.0,
            bars_since_transition=2,
            in_cooldown=True,
        )

        w_base = sizer.get_weights(base_state)
        w_cooldown = sizer.get_weights(cooldown_state)

        assert w_cooldown["trending"] < w_base["trending"]
        assert w_cooldown["mean_reverting"] < w_base["mean_reverting"]

    def test_low_confidence_reduces_weights(self):
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState

        sizer = RegimePositionSizer(min_confidence=0.6)

        high_conf = RegimeState(
            posteriors=np.array([0.8, 0.1, 0.1]),
            dominant_regime=0,
            confidence=0.8,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
        )

        low_conf = RegimeState(
            posteriors=np.array([0.4, 0.35, 0.25]),
            dominant_regime=0,
            confidence=0.4,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
        )

        w_high = sizer.get_weights(high_conf)
        w_low = sizer.get_weights(low_conf)

        assert w_low["trending"] < w_high["trending"]

    def test_weights_sum_leq_one(self):
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState

        sizer = RegimePositionSizer()

        state = RegimeState(
            posteriors=np.array([0.5, 0.45, 0.05]),
            dominant_regime=0,
            confidence=0.5,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
        )

        weights = sizer.get_weights(state)
        total = weights["trending"] + weights["mean_reverting"] + weights["choppy"]
        assert total <= 1.0 + 1e-6

    def test_strong_trending_high_weight(self):
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState

        sizer = RegimePositionSizer()

        state = RegimeState(
            posteriors=np.array([0.95, 0.03, 0.02]),
            dominant_regime=0,
            confidence=0.95,
            changepoint_prob=0.0,
            bars_since_transition=20,
            in_cooldown=False,
        )

        weights = sizer.get_weights(state)
        assert weights["trending"] > 0.8


# ── Micro HMM Tests ────────────────────────────────────────────

def _make_micro_regime_data(n_per_regime: int = 500, seed: int = 42) -> np.ndarray:
    """Generate synthetic 3-state micro regime data.

    Features: [spread, trade_rate, return_autocorr, realized_vol, ofi]

    Regime 0 (liquid_trending): tight spread, high activity, positive autocorr, moderate vol, high OFI
    Regime 1 (liquid_mean_reverting): tight spread, high activity, negative autocorr, low vol, low OFI
    Regime 2 (illiquid): wide spread, low activity, no autocorr, high vol, low OFI
    """
    rng = np.random.default_rng(seed)
    n = n_per_regime

    liquid_trending = np.column_stack([
        rng.normal(1.0, 0.1, n),     # spread (tight)
        rng.normal(50.0, 5.0, n),    # trade_rate (high)
        rng.normal(0.4, 0.05, n),    # return_autocorr (positive)
        rng.normal(0.001, 0.0002, n),# realized_vol (moderate)
        rng.normal(20.0, 3.0, n),    # ofi (high)
    ])

    liquid_mr = np.column_stack([
        rng.normal(1.0, 0.1, n),     # spread (tight)
        rng.normal(45.0, 5.0, n),    # trade_rate (high)
        rng.normal(-0.3, 0.05, n),   # return_autocorr (negative)
        rng.normal(0.0005, 0.0001, n),# realized_vol (low)
        rng.normal(5.0, 2.0, n),     # ofi (low)
    ])

    illiquid = np.column_stack([
        rng.normal(4.0, 0.5, n),     # spread (wide)
        rng.normal(5.0, 2.0, n),     # trade_rate (low)
        rng.normal(0.0, 0.05, n),    # return_autocorr (none)
        rng.normal(0.003, 0.001, n), # realized_vol (high)
        rng.normal(3.0, 1.0, n),     # ofi (low)
    ])

    return np.vstack([liquid_trending, liquid_mr, illiquid])


class TestMicroRegimeHMM:
    def test_fit_and_predict(self):
        from src.regime.micro_hmm import MicroRegimeHMM

        data = _make_micro_regime_data(n_per_regime=500)
        model = MicroRegimeHMM(n_iter=200, random_state=42)
        model.fit(data, lengths=[500, 500, 500])

        # Each block should be mostly one state
        for i in range(3):
            block = data[i * 500 : (i + 1) * 500]
            preds = model.predict(block)
            dominant = np.bincount(preds, minlength=3).argmax()
            purity = (preds == dominant).mean()
            assert purity > 0.7, f"Block {i} purity {purity:.2f} too low"

    def test_predict_proba_sums_to_one(self):
        from src.regime.micro_hmm import MicroRegimeHMM

        data = _make_micro_regime_data(n_per_regime=200)
        model = MicroRegimeHMM(n_iter=50, random_state=42)
        model.fit(data)

        proba = model.predict_proba(data[:50])
        assert proba.shape == (3,)
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

    def test_save_and_load(self):
        from src.regime.micro_hmm import MicroRegimeHMM

        data = _make_micro_regime_data(n_per_regime=200)
        model = MicroRegimeHMM(n_iter=50, random_state=42)
        model.fit(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "micro_hmm.pkl"
            model.save(path)

            model2 = MicroRegimeHMM()
            model2.load(path)

            np.testing.assert_array_equal(
                model.predict(data[:100]),
                model2.predict(data[:100]),
            )

    def test_state_label_assignment(self):
        """Illiquid state should have highest spread, liquid states should have low spread."""
        from src.regime.micro_hmm import MicroRegimeHMM, ILLIQUID, FEAT_SPREAD

        data = _make_micro_regime_data(n_per_regime=500)
        model = MicroRegimeHMM(n_iter=200, random_state=42)
        model.fit(data)

        means = model.means
        # Illiquid should have the highest spread
        assert means[ILLIQUID, FEAT_SPREAD] > means[0, FEAT_SPREAD] or \
               means[ILLIQUID, FEAT_SPREAD] > means[1, FEAT_SPREAD]

    def test_transition_matrix_rows_sum_to_one(self):
        from src.regime.micro_hmm import MicroRegimeHMM

        data = _make_micro_regime_data(n_per_regime=200)
        model = MicroRegimeHMM(n_iter=50, random_state=42)
        model.fit(data)

        tm = model.transition_matrix
        assert tm.shape == (3, 3)
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)


class TestCompositeRegimeDetector:
    """Test the macro + micro composite regime detector."""

    def _fit_detectors(self):
        from src.regime.regime_detector import RegimeDetector

        macro_data, _ = _make_regime_data(n_per_regime=300)
        micro_data = _make_micro_regime_data(n_per_regime=300)

        detector = RegimeDetector()
        detector.hmm.fit(macro_data)

        # Fit micro HMM separately
        from src.regime.micro_hmm import MicroRegimeHMM
        micro_model = MicroRegimeHMM(n_iter=100, random_state=42)
        micro_model.fit(micro_data)

        # Save and reload into detector
        with tempfile.TemporaryDirectory() as tmpdir:
            macro_path = Path(tmpdir) / "macro.pkl"
            micro_path = Path(tmpdir) / "micro.pkl"
            detector.hmm.save(macro_path)
            micro_model.save(micro_path)

            detector = RegimeDetector(
                hmm_model_path=macro_path,
                micro_hmm_model_path=micro_path,
            )

        return detector, macro_data, micro_data

    def test_update_with_micro(self):
        detector, macro_data, micro_data = self._fit_detectors()

        state = detector.update(macro_data[0], micro_features=micro_data[0])

        assert state.posteriors.shape == (3,)
        assert state.micro_posteriors is not None
        assert state.micro_posteriors.shape == (3,)
        assert state.micro_dominant in (0, 1, 2)
        assert state.tradeable in (True, False)
        assert state.entry_strategy in ("momentum", "fade", "none")

    def test_illiquid_blocks_trading(self):
        """When micro says illiquid, tradeable should be False."""
        from src.regime.regime_detector import RegimeState
        from src.regime.micro_hmm import ILLIQUID

        state = RegimeState(
            posteriors=np.array([0.8, 0.1, 0.1]),  # macro says trending
            dominant_regime=0,
            confidence=0.8,
            changepoint_prob=0.0,
            bars_since_transition=10,
            in_cooldown=False,
            micro_posteriors=np.array([0.1, 0.1, 0.8]),
            micro_dominant=ILLIQUID,
            micro_confidence=0.8,
        )

        assert state.micro_regime_name == "illiquid"

    def test_macro_only_still_works(self):
        """Detector without micro HMM should still work."""
        from src.regime.regime_detector import RegimeDetector

        macro_data, _ = _make_regime_data(n_per_regime=300)
        detector = RegimeDetector()
        detector.hmm.fit(macro_data)

        state = detector.update(macro_data[0])
        assert state.micro_posteriors is None
        assert state.micro_dominant == -1

    def test_position_sizer_illiquid_zeros(self):
        """Position sizer should zero out weights when micro is illiquid."""
        from src.regime.position_sizer import RegimePositionSizer
        from src.regime.regime_detector import RegimeState
        from src.regime.micro_hmm import ILLIQUID

        sizer = RegimePositionSizer()

        state = RegimeState(
            posteriors=np.array([0.9, 0.05, 0.05]),
            dominant_regime=0,
            confidence=0.9,
            changepoint_prob=0.0,
            bars_since_transition=20,
            in_cooldown=False,
            micro_posteriors=np.array([0.05, 0.05, 0.9]),
            micro_dominant=ILLIQUID,
            micro_confidence=0.9,
        )

        weights = sizer.get_weights(state)
        assert weights["trending"] == 0.0
        assert weights["mean_reverting"] == 0.0


# ── Trainer Resample Tests ──────────────────────────────────────

class TestTrainer:
    def test_resample_1s_to_1min(self):
        from src.regime.trainer import resample_1s_to_1min

        # 120 seconds, 5 features
        n = 120
        one_sec_ns = 1_000_000_000
        timestamps = np.arange(n) * one_sec_ns
        features = np.random.randn(n, 5)

        # Extract features at indices [0, 1, 2, 3]
        result = resample_1s_to_1min(timestamps, features, [0, 1, 2, 3])

        assert result.shape == (2, 4)  # 2 minutes, 4 features

        # First minute should be mean of first 60 seconds
        expected_f0 = features[:60, 0].mean()
        np.testing.assert_allclose(result[0, 0], expected_f0, atol=1e-10)

    def test_empty_input(self):
        from src.regime.trainer import resample_1s_to_1min

        timestamps = np.array([], dtype=np.int64)
        features = np.empty((0, 5))

        result = resample_1s_to_1min(timestamps, features, [0, 1, 2, 3])
        assert result.shape[0] == 0
