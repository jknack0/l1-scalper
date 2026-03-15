"""Combined HMM + BOCPD regime detector.

Processes 1-minute bar features and emits a RegimeState with posterior
probabilities, dominant regime, confidence, and changepoint detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.regime.bocpd import BOCPD
from src.regime.hmm import CHOPPY, MEAN_REVERTING, TRENDING, MarketRegimeHMM, STATE_NAMES

logger = logging.getLogger(__name__)

COOLDOWN_BARS = 5  # Bars to wait after a regime transition before trusting the new regime
CHANGEPOINT_THRESHOLD = 0.7


@dataclass
class RegimeState:
    """Current regime classification output."""

    posteriors: np.ndarray           # [P(trending), P(mean_reverting), P(choppy)]
    dominant_regime: int             # argmax of posteriors (0, 1, or 2)
    confidence: float                # max(posteriors)
    changepoint_prob: float          # from BOCPD [0, 1]
    bars_since_transition: int       # stability counter
    in_cooldown: bool                # True if recently transitioned

    @property
    def regime_name(self) -> str:
        return STATE_NAMES.get(self.dominant_regime, "unknown")


class RegimeDetector:
    """Combined HMM + BOCPD regime detector.

    Processes one 1-minute feature vector at a time and returns the
    current RegimeState with posteriors, confidence, and changepoint info.
    """

    def __init__(
        self,
        hmm_model_path: str | Path | None = None,
        cooldown_bars: int = COOLDOWN_BARS,
        changepoint_threshold: float = CHANGEPOINT_THRESHOLD,
    ) -> None:
        self._hmm = MarketRegimeHMM()
        if hmm_model_path is not None:
            self._hmm.load(hmm_model_path)

        self._bocpd = BOCPD()
        self._cooldown_bars = cooldown_bars
        self._changepoint_threshold = changepoint_threshold

        # State tracking
        self._history: list[np.ndarray] = []
        self._prev_regime: int = -1
        self._bars_since_transition: int = 0

    def update(self, one_min_features: np.ndarray) -> RegimeState:
        """Process one 1-minute bar and return current regime state.

        Args:
            one_min_features: shape [4] — [realized_vol, return_autocorr,
                                           spread_mean, trade_rate_mean]
        """
        assert one_min_features.shape == (4,), (
            f"Expected shape (4,), got {one_min_features.shape}"
        )

        self._history.append(one_min_features)

        # HMM posterior over the full observation sequence
        obs_array = np.array(self._history)
        posteriors = self._hmm.predict_proba(obs_array)

        dominant = int(np.argmax(posteriors))
        confidence = float(posteriors[dominant])

        # BOCPD on realized_vol (most sensitive to regime changes)
        cp_prob = self._bocpd.detect(float(one_min_features[0]))

        # Transition tracking
        if dominant != self._prev_regime:
            if self._prev_regime >= 0:
                logger.info(
                    "Regime transition: %s -> %s (confidence=%.2f, cp_prob=%.2f)",
                    STATE_NAMES.get(self._prev_regime, "?"),
                    STATE_NAMES.get(dominant, "?"),
                    confidence,
                    cp_prob,
                )
            self._prev_regime = dominant
            self._bars_since_transition = 0
        else:
            self._bars_since_transition += 1

        in_cooldown = self._bars_since_transition < self._cooldown_bars

        return RegimeState(
            posteriors=posteriors,
            dominant_regime=dominant,
            confidence=confidence,
            changepoint_prob=cp_prob,
            bars_since_transition=self._bars_since_transition,
            in_cooldown=in_cooldown,
        )

    def reset(self) -> None:
        """Reset for a new session."""
        self._history.clear()
        self._bocpd.reset()
        self._prev_regime = -1
        self._bars_since_transition = 0

    @property
    def hmm(self) -> MarketRegimeHMM:
        """Access the underlying HMM model (e.g., for fitting)."""
        return self._hmm
