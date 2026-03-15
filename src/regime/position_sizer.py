"""Converts regime posteriors to position size weights.

Choppy regime always gets weight 0 (we don't trade in chop).
Cooldown and low confidence reduce all weights.
"""

from __future__ import annotations

from src.regime.hmm import CHOPPY, MEAN_REVERTING, TRENDING, STATE_NAMES
from src.regime.regime_detector import RegimeState


class RegimePositionSizer:
    """Maps RegimeState to tradeable weights per regime.

    Returns {'trending': 0.0-1.0, 'mean_reverting': 0.0-1.0, 'choppy': 0.0}

    Logic:
        - Base weights = posteriors
        - Choppy weight always 0
        - Cooldown multiplier: 0.5 during cooldown
        - Low confidence: scale by confidence/min_confidence if below threshold
        - Normalize so trending + mean_reverting <= 1.0
    """

    def __init__(
        self,
        cooldown_bars: int = 5,
        min_confidence: float = 0.6,
    ) -> None:
        self._cooldown_bars = cooldown_bars
        self._min_confidence = min_confidence

    def get_weights(self, state: RegimeState) -> dict[str, float]:
        trending_w = float(state.posteriors[TRENDING])
        mr_w = float(state.posteriors[MEAN_REVERTING])
        choppy_w = 0.0  # Never trade in chop

        # Cooldown penalty
        if state.in_cooldown:
            trending_w *= 0.5
            mr_w *= 0.5

        # Low confidence penalty
        if state.confidence < self._min_confidence and self._min_confidence > 0:
            scale = state.confidence / self._min_confidence
            trending_w *= scale
            mr_w *= scale

        # Normalize so weights sum to <= 1.0
        total = trending_w + mr_w
        if total > 1.0:
            trending_w /= total
            mr_w /= total

        return {
            "trending": round(trending_w, 4),
            "mean_reverting": round(mr_w, 4),
            "choppy": choppy_w,
        }
