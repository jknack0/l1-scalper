"""3-state GMM-HMM for micro (local) regime classification.

Detects local market microstructure regimes at scalping timescales (30s-3min).
Uses 5 microstructure features computed from 1-second bars.

States:
    0 = LIQUID_TRENDING     — tight spread, active, directional order flow
    1 = LIQUID_MEAN_REVERTING — tight spread, active, balanced/oscillating
    2 = ILLIQUID            — wide spread, low activity, or no structure
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GMMHMM

from src.regime.micro_features import N_MICRO_FEATURES

logger = logging.getLogger(__name__)

# State labels
LIQUID_TRENDING = 0
LIQUID_MEAN_REVERTING = 1
ILLIQUID = 2

MICRO_STATE_NAMES = {
    LIQUID_TRENDING: "liquid_trending",
    LIQUID_MEAN_REVERTING: "liquid_mean_reverting",
    ILLIQUID: "illiquid",
}

# Feature indices
FEAT_SPREAD = 0
FEAT_TRADE_RATE = 1
FEAT_RETURN_AUTOCORR = 2
FEAT_REALIZED_VOL = 3
FEAT_OFI = 4


class MicroRegimeHMM:
    """3-state GMM-HMM for micro regime classification.

    Lower stickiness than the macro HMM (0.85 vs 0.95) since micro regimes
    transition faster. Uses 5 microstructure features.

    After fitting, state labels are mapped:
        - ILLIQUID = highest spread / lowest trade rate (illiquidity score)
        - LIQUID_TRENDING = higher return autocorrelation of remaining
        - LIQUID_MEAN_REVERTING = lower return autocorrelation of remaining
    """

    def __init__(
        self,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        n_mix: int = 2,
        sticky: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self._n_mix = n_mix
        self._sticky = sticky
        self._model = GMMHMM(
            n_components=3,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params="mcw",
            params="mcwt",  # learn transitions + emissions
        )
        self._model.startprob_ = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        off_diag = (1.0 - sticky) / 2.0
        self._model.transmat_ = np.array([
            [sticky, off_diag, off_diag],
            [off_diag, sticky, off_diag],
            [off_diag, off_diag, sticky],
        ])
        self._state_map: dict[int, int] = {0: 0, 1: 1, 2: 2}
        self._fitted = False

    def fit(self, data: np.ndarray, lengths: list[int] | None = None) -> None:
        """Fit the GMM-HMM on micro feature data. Shape: [n_bars, 5]."""
        assert data.ndim == 2 and data.shape[1] == N_MICRO_FEATURES, (
            f"Expected shape [n, {N_MICRO_FEATURES}], got {data.shape}"
        )

        logger.info("Fitting Micro GMM-HMM on %d windows (n_mix=%d, sticky=%.2f)",
                     data.shape[0], self._n_mix, self._sticky)

        self._model.fit(data, lengths=lengths)
        self._fitted = True

        self._assign_state_labels()

        logger.info(
            "Micro GMM-HMM fitted. Transition matrix:\n%s",
            np.array2string(self.transition_matrix, precision=3),
        )
        weighted_means = self._weighted_means()
        for raw, semantic in self._state_map.items():
            m = weighted_means[raw]
            logger.info(
                "  Raw state %d -> %s (spread=%.4f, trade_rate=%.4f, "
                "autocorr=%.4f, vol=%.6f, ofi=%.4f)",
                raw,
                MICRO_STATE_NAMES[semantic],
                m[FEAT_SPREAD],
                m[FEAT_TRADE_RATE],
                m[FEAT_RETURN_AUTOCORR],
                m[FEAT_REALIZED_VOL],
                m[FEAT_OFI],
            )

    def _weighted_means(self) -> np.ndarray:
        """Weight-averaged means per state. Returns [n_components, n_features]."""
        weights = self._model.weights_
        means = self._model.means_
        return np.einsum("cm,cmf->cf", weights, means)

    def _assign_state_labels(self) -> None:
        """Map raw HMM states to semantic labels.

        Heuristic:
            - ILLIQUID = highest illiquidity score (spread / trade_rate)
            - LIQUID_TRENDING = higher return autocorrelation of remaining two
            - LIQUID_MEAN_REVERTING = lower return autocorrelation of remaining two
        """
        wm = self._weighted_means()

        # Illiquidity score: high spread, low trade rate
        trade_rate = wm[:, FEAT_TRADE_RATE]
        spread = wm[:, FEAT_SPREAD]
        # Avoid division by zero
        safe_rate = np.maximum(trade_rate, 1e-10)
        illiquidity = spread / safe_rate

        illiquid_idx = int(np.argmax(illiquidity))

        # Of the remaining two, assign by return autocorrelation
        remaining = [i for i in range(3) if i != illiquid_idx]
        autocorrs = wm[remaining, FEAT_RETURN_AUTOCORR]
        if autocorrs[0] >= autocorrs[1]:
            trending_idx = remaining[0]
            mr_idx = remaining[1]
        else:
            trending_idx = remaining[1]
            mr_idx = remaining[0]

        self._state_map = {
            trending_idx: LIQUID_TRENDING,
            mr_idx: LIQUID_MEAN_REVERTING,
            illiquid_idx: ILLIQUID,
        }

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Return posterior probabilities for the last observation.

        Returns: shape [3] — [P(liquid_trending), P(liquid_mean_reverting), P(illiquid)]
        """
        assert self._fitted, "Model not fitted"
        raw_proba = self._model.predict_proba(observations)
        last_raw = raw_proba[-1]

        mapped = np.zeros(3)
        for raw_state, semantic_state in self._state_map.items():
            mapped[semantic_state] = last_raw[raw_state]
        return mapped

    def predict_proba_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Return posterior probabilities for every observation.

        Returns: shape [n_bars, 3]
        """
        assert self._fitted, "Model not fitted"
        raw_proba = self._model.predict_proba(observations)

        mapped = np.zeros((raw_proba.shape[0], 3))
        for raw_state, semantic_state in self._state_map.items():
            mapped[:, semantic_state] = raw_proba[:, raw_state]
        return mapped

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Return mapped state labels for every observation."""
        assert self._fitted, "Model not fitted"
        raw_labels = self._model.predict(observations)
        return np.array([self._state_map[int(s)] for s in raw_labels])

    @property
    def transition_matrix(self) -> np.ndarray:
        """Mapped transition matrix."""
        raw = self._model.transmat_
        mapped = np.zeros((3, 3))
        for ri in range(3):
            for rj in range(3):
                mapped[self._state_map[ri], self._state_map[rj]] = raw[ri, rj]
        return mapped

    @property
    def means(self) -> np.ndarray:
        """Mapped state means: shape [3, 5]."""
        raw = self._weighted_means()
        mapped = np.zeros_like(raw)
        for raw_state, semantic_state in self._state_map.items():
            mapped[semantic_state] = raw[raw_state]
        return mapped

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self._model, "state_map": self._state_map, "fitted": self._fitted},
                f,
            )
        logger.info("Saved Micro HMM to %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._state_map = data["state_map"]
        self._fitted = data["fitted"]
        logger.info("Loaded Micro HMM from %s", path)
