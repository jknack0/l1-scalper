"""3-state GMM-HMM for market regime classification.

Uses multiple Gaussian mixture components per state to capture fat-tailed
financial returns, with a sticky transition prior to encourage regime persistence.

States:
    0 = TRENDING       — persistent directional moves, positive return autocorrelation
    1 = MEAN_REVERTING — oscillating, negative return autocorrelation
    2 = CHOPPY         — noisy, no structure, low efficiency
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GMMHMM

logger = logging.getLogger(__name__)

# State labels
TRENDING = 0
MEAN_REVERTING = 1
CHOPPY = 2

STATE_NAMES = {TRENDING: "trending", MEAN_REVERTING: "mean_reverting", CHOPPY: "choppy"}

# Feature indices for the 4-feature input vector per 1-min bar
FEAT_RETURN_AUTOCORR = 0
FEAT_HURST = 1
FEAT_VARIANCE_RATIO = 2
FEAT_EFFICIENCY_RATIO = 3

N_FEATURES = 4


class MarketRegimeHMM:
    """3-state GMM-HMM for regime classification.

    Uses GMMHMM with multiple mixture components per state to handle
    fat-tailed financial returns, plus a sticky transition prior.

    Input: [n_bars, 4] array of features:
        [return_autocorr, hurst_exponent, variance_ratio, efficiency_ratio]

    After fitting, state labels are mapped so that:
        - TRENDING = state with highest mean hurst_exponent
        - MEAN_REVERTING = state with lowest mean hurst_exponent
        - CHOPPY = remaining state (low efficiency, hurst near 0.5)
    """

    def __init__(
        self,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        n_mix: int = 3,
        sticky: float = 0.95,
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
            init_params="mcw",  # init emissions only
            params="mcwt",      # learn transitions + emissions
        )
        # Sticky transition prior: high diagonal
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
        """Fit the GMM-HMM on feature data. Shape: [n_bars, 4].

        Transition matrix is fixed (sticky prior), only emissions are learned.
        """
        assert data.ndim == 2 and data.shape[1] == N_FEATURES, (
            f"Expected shape [n, {N_FEATURES}], got {data.shape}"
        )

        logger.info("Fitting GMM-HMM on %d bars (n_mix=%d, sticky=%.2f)",
                     data.shape[0], self._n_mix, self._sticky)

        self._model.fit(data, lengths=lengths)
        self._fitted = True

        # Map raw states to semantic labels
        self._assign_state_labels()

        logger.info(
            "GMM-HMM fitted. Transition matrix:\n%s",
            np.array2string(self.transition_matrix, precision=3),
        )
        weighted_means = self._weighted_means()
        for raw, semantic in self._state_map.items():
            m = weighted_means[raw]
            logger.info(
                "  Raw state %d -> %s (autocorr=%.4f, hurst=%.4f, var_ratio=%.4f, efficiency=%.4f)",
                raw,
                STATE_NAMES[semantic],
                m[FEAT_RETURN_AUTOCORR],
                m[FEAT_HURST],
                m[FEAT_VARIANCE_RATIO],
                m[FEAT_EFFICIENCY_RATIO],
            )

    def _weighted_means(self) -> np.ndarray:
        """Compute weight-averaged means per state from GMMHMM.

        GMMHMM stores means_ as [n_components, n_mix, n_features]
        and weights_ as [n_components, n_mix]. Returns [n_components, n_features].
        """
        # means_: [n_components, n_mix, n_features], weights_: [n_components, n_mix]
        weights = self._model.weights_  # [n_components, n_mix]
        means = self._model.means_     # [n_components, n_mix, n_features]
        # Weighted average across mixture components
        return np.einsum("cm,cmf->cf", weights, means)

    def _assign_state_labels(self) -> None:
        """Map raw HMM states to semantic labels.

        Heuristic using Hurst exponent as primary discriminator:
            - TRENDING = highest mean Hurst (H > 0.5, persistent)
            - MEAN_REVERTING = lowest mean Hurst (H < 0.5, anti-persistent)
            - CHOPPY = remaining (H ≈ 0.5, random walk / no structure)
        """
        wm = self._weighted_means()
        hursts = wm[:, FEAT_HURST]
        sorted_idx = np.argsort(hursts)

        # Lowest Hurst = mean-reverting, highest = trending, middle = choppy
        self._state_map = {
            int(sorted_idx[0]): MEAN_REVERTING,
            int(sorted_idx[1]): CHOPPY,
            int(sorted_idx[2]): TRENDING,
        }

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Return posterior probabilities for the last observation.

        Args:
            observations: shape [n_bars, 4] — full sequence seen so far

        Returns:
            shape [3] — [P(trending), P(mean_reverting), P(choppy)]
        """
        assert self._fitted, "Model not fitted"
        raw_proba = self._model.predict_proba(observations)
        last_raw = raw_proba[-1]  # posteriors for the latest bar

        # Remap to semantic ordering
        mapped = np.zeros(3)
        for raw_state, semantic_state in self._state_map.items():
            mapped[semantic_state] = last_raw[raw_state]

        return mapped

    def predict_proba_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Return posterior probabilities for every observation.

        Returns:
            shape [n_bars, 3] — each row is [P(trending), P(mean_reverting), P(choppy)]
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
        """Mapped transition matrix: [from_state, to_state]."""
        raw = self._model.transmat_
        n = 3
        mapped = np.zeros((n, n))
        for ri in range(n):
            for rj in range(n):
                si = self._state_map[ri]
                sj = self._state_map[rj]
                mapped[si, sj] = raw[ri, rj]
        return mapped

    @property
    def means(self) -> np.ndarray:
        """Mapped state means: shape [3, 4] (weighted across mixture components)."""
        raw = self._weighted_means()
        mapped = np.zeros_like(raw)
        for raw_state, semantic_state in self._state_map.items():
            mapped[semantic_state] = raw[raw_state]
        return mapped

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self._model, "state_map": self._state_map, "fitted": self._fitted},
                f,
            )
        logger.info("Saved HMM to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._state_map = data["state_map"]
        self._fitted = data["fitted"]
        logger.info("Loaded HMM from %s", path)
