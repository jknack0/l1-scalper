"""Converts regime posteriors to position size weights.

Uses both macro and micro regime states:
    - Macro CHOPPY or Micro ILLIQUID → weight 0 (no trading)
    - Cooldown and low confidence reduce all weights
    - Micro state influences strategy selection (momentum vs fade)
"""

from __future__ import annotations

from src.regime.hmm import CHOPPY, MEAN_REVERTING, TRENDING
from src.regime.micro_hmm import ILLIQUID, LIQUID_MEAN_REVERTING, LIQUID_TRENDING
from src.regime.regime_detector import RegimeState


class RegimePositionSizer:
    """Maps RegimeState to tradeable weights.

    Returns {'trending': 0.0-1.0, 'mean_reverting': 0.0-1.0, 'choppy': 0.0}

    Logic:
        - Base weights = macro posteriors
        - Choppy weight always 0
        - Micro ILLIQUID zeros out all weights
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

        # Micro ILLIQUID gate — zero everything
        if state.micro_dominant == ILLIQUID:
            return {
                "trending": 0.0,
                "mean_reverting": 0.0,
                "choppy": 0.0,
            }

        # Macro cooldown penalty
        if state.in_cooldown:
            trending_w *= 0.5
            mr_w *= 0.5

        # Micro cooldown penalty (additional reduction)
        if state.micro_in_cooldown:
            trending_w *= 0.7
            mr_w *= 0.7

        # Low macro confidence penalty
        if state.confidence < self._min_confidence and self._min_confidence > 0:
            scale = state.confidence / self._min_confidence
            trending_w *= scale
            mr_w *= scale

        # Low micro confidence penalty
        if state.micro_confidence > 0 and state.micro_confidence < self._min_confidence:
            scale = state.micro_confidence / self._min_confidence
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
