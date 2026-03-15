"""3-state Gaussian HMM for market regime classification.

States:
    0 = TRENDING       — persistent directional moves, positive return autocorrelation
    1 = MEAN_REVERTING — oscillating, negative return autocorrelation
    2 = CHOPPY         — noisy, no structure, high spread, low trade rate
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# State labels
TRENDING = 0
MEAN_REVERTING = 1
CHOPPY = 2

STATE_NAMES = {TRENDING: "trending", MEAN_REVERTING: "mean_reverting", CHOPPY: "choppy"}

# Feature indices for the 4-feature input vector per 1-min bar
FEAT_REALIZED_VOL = 0
FEAT_RETURN_AUTOCORR = 1
FEAT_SPREAD_MEAN = 2
FEAT_TRADE_RATE_MEAN = 3

N_FEATURES = 4


class MarketRegimeHMM:
    """3-state Gaussian HMM for regime classification.

    Input: [n_bars, 4] array of 1-minute bar features:
        [realized_vol, return_autocorr, spread_mean, trade_rate_mean]

    After fitting, state labels are mapped so that:
        - TRENDING = state with highest mean return_autocorr
        - MEAN_REVERTING = state with lowest mean return_autocorr
        - CHOPPY = remaining state
    """

    def __init__(
        self,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self._model = GaussianHMM(
            n_components=3,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
        self._state_map: dict[int, int] = {0: 0, 1: 1, 2: 2}
        self._fitted = False

    def fit(self, data: np.ndarray, lengths: list[int] | None = None) -> None:
        """Fit the HMM on 1-minute bar features. Shape: [n_bars, 4].

        Uses GMM to initialize emission parameters before EM, which avoids
        the local-optima problem where HMM merges two distinct regimes.
        """
        assert data.ndim == 2 and data.shape[1] == N_FEATURES, (
            f"Expected shape [n, {N_FEATURES}], got {data.shape}"
        )

        logger.info("Fitting HMM on %d bars", data.shape[0])

        # Initialize with GMM for better emission estimates
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=10, random_state=42)
        gmm.fit(data)
        self._model.means_ = gmm.means_
        self._model.covars_ = gmm.covariances_

        # Only fit transitions and start probs from EM, keep GMM emissions as init
        self._model.init_params = "st"
        self._model.fit(data, lengths=lengths)
        self._fitted = True

        # Map raw states to semantic labels based on return_autocorr means
        self._assign_state_labels()

        logger.info(
            "HMM fitted. Transition matrix:\n%s",
            np.array2string(self.transition_matrix, precision=3),
        )
        for raw, semantic in self._state_map.items():
            means = self._model.means_[raw]
            logger.info(
                "  Raw state %d -> %s (vol=%.4f, autocorr=%.4f, spread=%.4f, rate=%.4f)",
                raw,
                STATE_NAMES[semantic],
                means[FEAT_REALIZED_VOL],
                means[FEAT_RETURN_AUTOCORR],
                means[FEAT_SPREAD_MEAN],
                means[FEAT_TRADE_RATE_MEAN],
            )

    def _assign_state_labels(self) -> None:
        """Map raw HMM states to semantic labels.

        Heuristic:
            - TRENDING = highest absolute return_autocorr (positive)
            - MEAN_REVERTING = most negative return_autocorr
            - CHOPPY = remaining (highest spread or lowest trade rate)
        """
        autocorrs = self._model.means_[:, FEAT_RETURN_AUTOCORR]
        sorted_idx = np.argsort(autocorrs)

        # Lowest autocorr = mean-reverting, highest = trending, middle = choppy
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
        """Mapped state means: shape [3, 4]."""
        raw = self._model.means_
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
