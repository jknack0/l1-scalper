"""Combined macro + micro regime detector.

Macro HMM (1-min features): session-level regime — trending / mean-reverting / choppy.
Micro HMM (1-sec features): local regime — liquid-trending / liquid-mean-reverting / illiquid.

Gating logic:
    - Macro CHOPPY → never trade
    - Macro OK + Micro ILLIQUID → don't trade
    - Macro OK + Micro LIQUID → trade, micro state selects momentum vs fade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.regime.bocpd import BOCPD
from src.regime.hmm import CHOPPY, MEAN_REVERTING, TRENDING, MarketRegimeHMM, STATE_NAMES
from src.regime.micro_hmm import (
    ILLIQUID,
    LIQUID_MEAN_REVERTING,
    LIQUID_TRENDING,
    MICRO_STATE_NAMES,
    MicroRegimeHMM,
)

logger = logging.getLogger(__name__)

COOLDOWN_BARS = 5
CHANGEPOINT_THRESHOLD = 0.7
MICRO_COOLDOWN_BARS = 3  # Faster cooldown for micro regimes


@dataclass
class RegimeState:
    """Current regime classification output (macro + micro)."""

    # Macro regime
    posteriors: np.ndarray           # [P(trending), P(mean_reverting), P(choppy)]
    dominant_regime: int             # argmax of posteriors (0, 1, or 2)
    confidence: float                # max(posteriors)
    changepoint_prob: float          # from BOCPD [0, 1]
    bars_since_transition: int       # stability counter
    in_cooldown: bool                # True if recently transitioned

    # Micro regime (None if micro detector not loaded)
    micro_posteriors: np.ndarray | None = field(default=None)
    micro_dominant: int = -1
    micro_confidence: float = 0.0
    micro_bars_since_transition: int = 0
    micro_in_cooldown: bool = False

    # Composite gate
    tradeable: bool = False
    entry_strategy: str = "none"     # "momentum" | "fade" | "none"

    @property
    def regime_name(self) -> str:
        return STATE_NAMES.get(self.dominant_regime, "unknown")

    @property
    def micro_regime_name(self) -> str:
        return MICRO_STATE_NAMES.get(self.micro_dominant, "unknown")


class RegimeDetector:
    """Combined macro + micro regime detector.

    Macro: processes 1-minute feature vectors, detects session-level regimes.
    Micro: processes 1-second feature windows, detects local tradeable conditions.
    """

    def __init__(
        self,
        hmm_model_path: str | Path | None = None,
        micro_hmm_model_path: str | Path | None = None,
        cooldown_bars: int = COOLDOWN_BARS,
        micro_cooldown_bars: int = MICRO_COOLDOWN_BARS,
        changepoint_threshold: float = CHANGEPOINT_THRESHOLD,
        min_macro_confidence: float = 0.5,
        min_micro_confidence: float = 0.5,
    ) -> None:
        # Macro HMM
        self._hmm = MarketRegimeHMM()
        if hmm_model_path is not None:
            self._hmm.load(hmm_model_path)

        # Micro HMM (optional)
        self._micro_hmm: MicroRegimeHMM | None = None
        if micro_hmm_model_path is not None:
            self._micro_hmm = MicroRegimeHMM()
            self._micro_hmm.load(micro_hmm_model_path)

        self._bocpd = BOCPD()
        self._cooldown_bars = cooldown_bars
        self._micro_cooldown_bars = micro_cooldown_bars
        self._changepoint_threshold = changepoint_threshold
        self._min_macro_confidence = min_macro_confidence
        self._min_micro_confidence = min_micro_confidence

        # Macro state tracking
        self._history: list[np.ndarray] = []
        self._prev_regime: int = -1
        self._bars_since_transition: int = 0

        # Micro state tracking
        self._micro_history: list[np.ndarray] = []
        self._prev_micro_regime: int = -1
        self._micro_bars_since_transition: int = 0

    def update(
        self,
        one_min_features: np.ndarray,
        micro_features: np.ndarray | None = None,
    ) -> RegimeState:
        """Process one bar and return current regime state.

        Args:
            one_min_features: shape [4] — macro HMM features
            micro_features: shape [5] — micro HMM features (optional)
        """
        assert one_min_features.shape == (4,), (
            f"Expected macro shape (4,), got {one_min_features.shape}"
        )

        # ── Macro update ────────────────────────────────────────
        self._history.append(one_min_features)
        obs_array = np.array(self._history)
        posteriors = self._hmm.predict_proba(obs_array)

        dominant = int(np.argmax(posteriors))
        confidence = float(posteriors[dominant])

        cp_prob = self._bocpd.detect(float(one_min_features[3]))

        if dominant != self._prev_regime:
            if self._prev_regime >= 0:
                logger.info(
                    "Macro regime transition: %s -> %s (confidence=%.2f, cp_prob=%.2f)",
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

        # ── Micro update ────────────────────────────────────────
        micro_posteriors = None
        micro_dominant = -1
        micro_confidence = 0.0
        micro_in_cooldown = False

        if micro_features is not None and self._micro_hmm is not None:
            assert micro_features.shape == (5,), (
                f"Expected micro shape (5,), got {micro_features.shape}"
            )
            self._micro_history.append(micro_features)
            micro_obs = np.array(self._micro_history)
            micro_posteriors = self._micro_hmm.predict_proba(micro_obs)

            micro_dominant = int(np.argmax(micro_posteriors))
            micro_confidence = float(micro_posteriors[micro_dominant])

            if micro_dominant != self._prev_micro_regime:
                if self._prev_micro_regime >= 0:
                    logger.info(
                        "Micro regime transition: %s -> %s (confidence=%.2f)",
                        MICRO_STATE_NAMES.get(self._prev_micro_regime, "?"),
                        MICRO_STATE_NAMES.get(micro_dominant, "?"),
                        micro_confidence,
                    )
                self._prev_micro_regime = micro_dominant
                self._micro_bars_since_transition = 0
            else:
                self._micro_bars_since_transition += 1

            micro_in_cooldown = self._micro_bars_since_transition < self._micro_cooldown_bars

        # ── Composite gate ──────────────────────────────────────
        tradeable, entry_strategy = self._compute_gate(
            dominant, confidence, in_cooldown,
            micro_dominant, micro_confidence, micro_in_cooldown,
        )

        return RegimeState(
            posteriors=posteriors,
            dominant_regime=dominant,
            confidence=confidence,
            changepoint_prob=cp_prob,
            bars_since_transition=self._bars_since_transition,
            in_cooldown=in_cooldown,
            micro_posteriors=micro_posteriors,
            micro_dominant=micro_dominant,
            micro_confidence=micro_confidence,
            micro_bars_since_transition=self._micro_bars_since_transition,
            micro_in_cooldown=micro_in_cooldown,
            tradeable=tradeable,
            entry_strategy=entry_strategy,
        )

    def _compute_gate(
        self,
        macro_regime: int,
        macro_confidence: float,
        macro_cooldown: bool,
        micro_regime: int,
        micro_confidence: float,
        micro_cooldown: bool,
    ) -> tuple[bool, str]:
        """Determine if trading is allowed and which strategy to use."""
        # Macro CHOPPY → never trade
        if macro_regime == CHOPPY:
            return False, "none"

        # Macro cooldown or low confidence → don't trade
        if macro_cooldown or macro_confidence < self._min_macro_confidence:
            return False, "none"

        # If no micro detector loaded, use macro only
        if self._micro_hmm is None or micro_regime < 0:
            strategy = "momentum" if macro_regime == TRENDING else "fade"
            return True, strategy

        # Micro ILLIQUID → don't trade
        if micro_regime == ILLIQUID:
            return False, "none"

        # Micro cooldown or low confidence → don't trade
        if micro_cooldown or micro_confidence < self._min_micro_confidence:
            return False, "none"

        # Both macro and micro say go — pick strategy from micro state
        if micro_regime == LIQUID_TRENDING:
            return True, "momentum"
        else:  # LIQUID_MEAN_REVERTING
            return True, "fade"

    def reset(self) -> None:
        """Reset for a new session."""
        self._history.clear()
        self._micro_history.clear()
        self._bocpd.reset()
        self._prev_regime = -1
        self._bars_since_transition = 0
        self._prev_micro_regime = -1
        self._micro_bars_since_transition = 0

    @property
    def hmm(self) -> MarketRegimeHMM:
        return self._hmm

    @property
    def micro_hmm(self) -> MicroRegimeHMM | None:
        return self._micro_hmm
