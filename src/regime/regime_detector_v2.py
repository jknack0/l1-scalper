"""Combined macro + micro regime detector (v2).

Uses v2 HMMs with forward-only incremental inference and RegimePairRouter
for (macro, micro) -> model_id routing. No heuristic state labels.

Key differences from v1:
    - Forward-only incremental inference (no growing history buffer)
    - RegimePairRouter for regime pair -> entry model mapping
    - No hardcoded state semantics (trending/MR/choppy)
    - Normalization using training stats stored in HMM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.regime.macro_hmm_v2 import MacroRegimeHMMv2
from src.regime.micro_hmm_v2 import MicroRegimeHMMv2
from src.regime.regime_pair import RegimePairRouter

logger = logging.getLogger(__name__)

COOLDOWN_BARS = 5
MICRO_COOLDOWN_BARS = 3


@dataclass
class RegimeStateV2:
    """Current regime classification output (v2)."""

    # Macro regime
    macro_posteriors: np.ndarray     # [n_macro_states]
    macro_state: int                 # argmax
    macro_confidence: float          # max(posteriors)
    macro_bars_since_transition: int
    macro_in_cooldown: bool

    # Micro regime
    micro_posteriors: np.ndarray     # [n_micro_states]
    micro_state: int                 # argmax
    micro_confidence: float
    micro_bars_since_transition: int
    micro_in_cooldown: bool

    # Routing result
    tradeable: bool
    model_id: str                    # which entry model to use


class RegimeDetectorV2:
    """Combined macro + micro regime detector with pair routing.

    Uses incremental forward filtering (O(1) per update, not O(n)).
    Routes regime pairs to entry models via RegimePairRouter.
    """

    def __init__(
        self,
        macro_hmm_path: str | Path,
        micro_hmm_path: str | Path,
        router_path: str | Path,
        cooldown_bars: int = COOLDOWN_BARS,
        micro_cooldown_bars: int = MICRO_COOLDOWN_BARS,
        min_macro_confidence: float = 0.5,
        min_micro_confidence: float = 0.5,
    ) -> None:
        # Load v2 HMMs
        self._macro_hmm = MacroRegimeHMMv2()
        self._macro_hmm.load(macro_hmm_path)

        self._micro_hmm = MicroRegimeHMMv2()
        self._micro_hmm.load(micro_hmm_path)

        # Load pair router
        self._router = RegimePairRouter.load(router_path)

        self._cooldown_bars = cooldown_bars
        self._micro_cooldown_bars = micro_cooldown_bars
        self._min_macro_confidence = min_macro_confidence
        self._min_micro_confidence = min_micro_confidence

        # State tracking
        self._prev_macro: int = -1
        self._macro_bars_since: int = 0
        self._prev_micro: int = -1
        self._micro_bars_since: int = 0

    def update(
        self,
        macro_features: np.ndarray,
        micro_features: np.ndarray,
    ) -> RegimeStateV2:
        """Process one macro + micro observation and return regime state.

        Args:
            macro_features: shape [n_macro_features] — raw (unnormalized).
            micro_features: shape [n_micro_features] — raw (unnormalized).

        Returns:
            RegimeStateV2 with routing decision.
        """
        # Normalize using training stats stored in HMMs
        macro_norm = self._macro_hmm.normalize(macro_features.reshape(1, -1))[0]
        micro_norm = self._micro_hmm.normalize(micro_features.reshape(1, -1))[0]

        # Incremental forward filtering (O(1) per call)
        macro_post = self._macro_hmm.predict_proba_incremental(macro_norm)
        micro_post = self._micro_hmm.predict_proba_incremental(micro_norm)

        macro_state = int(np.argmax(macro_post))
        macro_conf = float(macro_post[macro_state])
        micro_state = int(np.argmax(micro_post))
        micro_conf = float(micro_post[micro_state])

        # Track transitions
        if macro_state != self._prev_macro:
            if self._prev_macro >= 0:
                logger.info("Macro transition: %d -> %d (conf=%.2f)",
                           self._prev_macro, macro_state, macro_conf)
            self._prev_macro = macro_state
            self._macro_bars_since = 0
        else:
            self._macro_bars_since += 1

        if micro_state != self._prev_micro:
            if self._prev_micro >= 0:
                logger.info("Micro transition: %d -> %d (conf=%.2f)",
                           self._prev_micro, micro_state, micro_conf)
            self._prev_micro = micro_state
            self._micro_bars_since = 0
        else:
            self._micro_bars_since += 1

        macro_cooldown = self._macro_bars_since < self._cooldown_bars
        micro_cooldown = self._micro_bars_since < self._micro_cooldown_bars

        # Routing decision
        tradeable, model_id = self._compute_gate(
            macro_state, macro_conf, macro_cooldown,
            micro_state, micro_conf, micro_cooldown,
        )

        return RegimeStateV2(
            macro_posteriors=macro_post,
            macro_state=macro_state,
            macro_confidence=macro_conf,
            macro_bars_since_transition=self._macro_bars_since,
            macro_in_cooldown=macro_cooldown,
            micro_posteriors=micro_post,
            micro_state=micro_state,
            micro_confidence=micro_conf,
            micro_bars_since_transition=self._micro_bars_since,
            micro_in_cooldown=micro_cooldown,
            tradeable=tradeable,
            model_id=model_id,
        )

    def _compute_gate(
        self,
        macro_state: int,
        macro_conf: float,
        macro_cooldown: bool,
        micro_state: int,
        micro_conf: float,
        micro_cooldown: bool,
    ) -> tuple[bool, str]:
        """Determine tradability and route to model."""
        # Confidence and cooldown gates
        if macro_cooldown or macro_conf < self._min_macro_confidence:
            return False, "skip"
        if micro_cooldown or micro_conf < self._min_micro_confidence:
            return False, "skip"

        # Route through pair router
        model_id, tradeable = self._router.route(macro_state, micro_state)
        return tradeable, model_id

    def reset(self) -> None:
        """Reset for a new session. Call at RTH open."""
        self._macro_hmm.reset_filter()
        self._micro_hmm.reset_filter()
        self._prev_macro = -1
        self._macro_bars_since = 0
        self._prev_micro = -1
        self._micro_bars_since = 0

    @property
    def macro_hmm(self) -> MacroRegimeHMMv2:
        return self._macro_hmm

    @property
    def micro_hmm(self) -> MicroRegimeHMMv2:
        return self._micro_hmm

    @property
    def router(self) -> RegimePairRouter:
        return self._router
