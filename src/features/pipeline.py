"""FeaturePipeline: orchestrates all features, resamples to 1s, outputs tensor."""

from __future__ import annotations

from collections import deque

import numpy as np

from src.data.schemas import L1Record
from src.features.base import Feature
from src.features.normalizer import WelfordNormalizer, staleness_gate

ONE_SEC_NS = 1_000_000_000


class FeaturePipeline:
    """Orchestrates L1 features into normalized [window_size, num_features] tensors.

    - Resamples to 1-second bars (takes last value per second for each feature)
    - Normalizes each feature independently via WelfordNormalizer
    - Applies staleness check (zeros out stale features)
    - Maintains a circular buffer of window_size normalized feature vectors
    """

    def __init__(
        self,
        features: list[Feature],
        window_size: int = 100,
        normalizer_window: int = 100,
        staleness_ns: int = 2_000_000_000,
    ) -> None:
        self._features = features
        self._window_size = window_size
        self._staleness_ns = staleness_ns
        self._num_features = len(features)

        # Per-feature state
        self._normalizers = [WelfordNormalizer(normalizer_window) for _ in features]
        self._last_values = [0.0] * self._num_features
        self._last_update_ts = [0] * self._num_features
        self._feature_ready = [False] * self._num_features

        # 1-second resampling state
        self._current_second: int = 0
        self._current_second_values: list[float] = [0.0] * self._num_features

        # Rolling window buffer
        self._window: deque[np.ndarray] = deque(maxlen=window_size)

    def process_record(self, record: L1Record) -> np.ndarray | None:
        """Process one record. Returns normalized feature vector if a new 1s bar completed."""
        ts = record.timestamp
        second = ts // ONE_SEC_NS

        # Update all features
        for i, feature in enumerate(self._features):
            val = feature.update(record)
            if val is not None:
                self._last_values[i] = val
                self._last_update_ts[i] = ts
                self._feature_ready[i] = True

        # Build current second's values with staleness gate
        for i in range(self._num_features):
            gated = staleness_gate(
                self._last_values[i], ts, self._last_update_ts[i], self._staleness_ns
            )
            self._current_second_values[i] = gated

        # Check for second boundary
        if second != self._current_second and self._current_second > 0:
            # Emit the completed second
            vec = self._emit_second(ts)
            self._current_second = second
            return vec

        if self._current_second == 0:
            self._current_second = second

        return None

    def _emit_second(self, ts: int) -> np.ndarray | None:
        """Normalize and buffer the completed 1-second feature vector."""
        normalized = np.zeros(self._num_features, dtype=np.float64)

        for i in range(self._num_features):
            raw = self._current_second_values[i]
            normalized[i] = self._normalizers[i].update(raw)

        self._window.append(normalized)
        return normalized

    def get_window(self) -> np.ndarray | None:
        """Returns shape [window_size, num_features] if window is full, else None."""
        if len(self._window) < self._window_size:
            return None
        return np.array(self._window)

    def is_ready(self) -> bool:
        """True if window is full and all features have produced at least one value."""
        return (
            len(self._window) >= self._window_size
            and all(self._feature_ready)
        )

    def reset_session(self) -> None:
        """Reset all features and buffers for a new RTH session."""
        for f in self._features:
            f.reset()
        for n in self._normalizers:
            n.__init__(n._window)  # type: ignore[misc]

        self._last_values = [0.0] * self._num_features
        self._last_update_ts = [0] * self._num_features
        self._feature_ready = [False] * self._num_features
        self._current_second = 0
        self._current_second_values = [0.0] * self._num_features
        self._window.clear()


def build_default_pipeline(window_size: int = 100) -> FeaturePipeline:
    """Build the full 12-feature L1 pipeline."""
    from src.features.cvd import CumulativeVolumeDelta
    from src.features.hurst import HurstExponent
    from src.features.lee_ready import LeeReadyClassifier
    from src.features.microprice import MicroPrice
    from src.features.ofi import OrderFlowImbalance
    from src.features.realized_vol import RealizedVolatility
    from src.features.return_autocorr import ReturnAutocorrelation
    from src.features.spread import Spread
    from src.features.trade_rate import TradeRate
    from src.features.trade_size_dist import TradeSizeDistribution
    from src.features.volume_profile import VolumeProfile
    from src.features.vpin import VPIN

    features: list[Feature] = [
        OrderFlowImbalance(),      # 0: ofi
        VPIN(),                    # 1: vpin
        MicroPrice(),              # 2: microprice
        CumulativeVolumeDelta(),   # 3: cvd
        Spread(),                  # 4: spread
        TradeRate(),               # 5: trade_rate
        VolumeProfile(),           # 6: volume_profile (POC distance)
        HurstExponent(),           # 7: hurst
        RealizedVolatility(),      # 8: realized_vol
        LeeReadyClassifier(),      # 9: lee_ready
        TradeSizeDistribution(),   # 10: trade_size_dist
        ReturnAutocorrelation(),   # 11: return_autocorr
    ]

    return FeaturePipeline(features, window_size=window_size)
