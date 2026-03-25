"""N-state GMM-HMM for micro (local) regime classification (v2).

Same architecture as MacroRegimeHMMv2 but with different defaults:
    - Lower sticky prior (0.85) for faster micro transitions
    - 2 mixture components (lighter model for 30-sec resolution)
    - BIC selection over N=3,4,5

States are unlabeled — meaning determined by regime validation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GMMHMM

from src.regime.macro_hmm_v2 import HMMFitResult, _gmmhmm_n_params, _logsumexp

logger = logging.getLogger(__name__)


class MicroRegimeHMMv2:
    """N-state GMM-HMM for micro regime classification.

    Same forward-only inference and BIC selection as MacroRegimeHMMv2,
    tuned for faster-transitioning microstructure regimes.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_mix: int = 2,
        sticky: float = 0.85,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self._n_states = n_states
        self._n_mix = n_mix
        self._sticky = sticky
        self._covariance_type = covariance_type
        self._n_iter = n_iter
        self._tol = tol
        self._random_state = random_state

        self._model: GMMHMM | None = None
        self._fitted = False

        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

        self._alpha: np.ndarray | None = None

    def _create_model(self, n_states: int) -> GMMHMM:
        model = GMMHMM(
            n_components=n_states,
            n_mix=self._n_mix,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            tol=self._tol,
            random_state=self._random_state,
            init_params="mcw",
            params="mcwt",
        )
        model.startprob_ = np.ones(n_states) / n_states
        off_diag = (1.0 - self._sticky) / (n_states - 1)
        transmat = np.full((n_states, n_states), off_diag)
        np.fill_diagonal(transmat, self._sticky)
        model.transmat_ = transmat
        return model

    def fit(
        self,
        data: np.ndarray,
        lengths: list[int] | None = None,
    ) -> None:
        """Fit on raw (unnormalized) features. Stores training stats."""
        assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"

        self._train_mean = data.mean(axis=0)
        self._train_std = data.std(axis=0)
        self._train_std[self._train_std < 1e-12] = 1.0

        data_norm = (data - self._train_mean) / self._train_std

        logger.info(
            "Fitting MicroHMMv2: n_states=%d, n_mix=%d, sticky=%.2f, %d windows",
            self._n_states, self._n_mix, self._sticky, data_norm.shape[0],
        )

        self._model = self._create_model(self._n_states)
        self._model.fit(data_norm, lengths=lengths)
        self._fitted = True

        logger.info("Transition matrix:\n%s",
                     np.array2string(self._model.transmat_, precision=3))
        self._log_state_distributions(data_norm, lengths)

    def fit_with_bic(
        self,
        data: np.ndarray,
        lengths: list[int] | None = None,
        n_states_candidates: list[int] | None = None,
    ) -> list[HMMFitResult]:
        """Fit multiple candidates, select by BIC."""
        if n_states_candidates is None:
            n_states_candidates = [3, 4, 5]

        self._train_mean = data.mean(axis=0)
        self._train_std = data.std(axis=0)
        self._train_std[self._train_std < 1e-12] = 1.0
        data_norm = (data - self._train_mean) / self._train_std

        n_samples = data_norm.shape[0]
        results: list[HMMFitResult] = []
        best_bic = float("inf")
        best_model = None
        best_n = -1

        for n in n_states_candidates:
            logger.info("Fitting candidate: n_states=%d", n)
            model = self._create_model(n)

            try:
                model.fit(data_norm, lengths=lengths)
                ll = model.score(data_norm, lengths=lengths)

                n_feat = data_norm.shape[1]
                k = _gmmhmm_n_params(n, self._n_mix, n_feat, self._covariance_type)
                bic = -2 * ll + k * np.log(n_samples)
                aic = -2 * ll + 2 * k

                result = HMMFitResult(
                    n_states=n, bic=bic, aic=aic,
                    log_likelihood=ll, converged=True,
                )
                results.append(result)

                logger.info(
                    "  n_states=%d: BIC=%.1f, AIC=%.1f, LL=%.1f, params=%d",
                    n, bic, aic, ll, k,
                )

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_n = n

            except Exception as e:
                logger.warning("  n_states=%d: FAILED (%s)", n, e)
                results.append(HMMFitResult(
                    n_states=n, bic=float("inf"), aic=float("inf"),
                    log_likelihood=float("-inf"), converged=False,
                ))

        if best_model is None:
            raise RuntimeError("All GMMHMM candidates failed to fit")

        self._model = best_model
        self._n_states = best_n
        self._fitted = True

        logger.info("Selected n_states=%d (BIC=%.1f)", best_n, best_bic)
        logger.info("Transition matrix:\n%s",
                     np.array2string(self._model.transmat_, precision=3))
        self._log_state_distributions(data_norm, lengths)

        return results

    def _log_state_distributions(
        self,
        data_norm: np.ndarray,
        lengths: list[int] | None = None,
    ) -> None:
        labels = self._model.predict(data_norm, lengths=lengths)
        for s in range(self._n_states):
            count = (labels == s).sum()
            pct = count / len(labels) * 100
            logger.info("  State %d: %d windows (%.1f%%)", s, count, pct)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        assert self._train_mean is not None, "Model not fitted"
        return (data - self._train_mean) / self._train_std

    def predict_proba_forward(self, observations: np.ndarray) -> np.ndarray:
        """Forward-only filtering. Returns [n_obs, n_states] posteriors."""
        assert self._fitted, "Model not fitted"
        n_obs = observations.shape[0]
        n_states = self._n_states

        log_emiss = self._model._compute_log_likelihood(observations)
        log_trans = np.log(self._model.transmat_ + 1e-300)
        log_start = np.log(self._model.startprob_ + 1e-300)

        posteriors = np.zeros((n_obs, n_states))

        log_alpha = log_start + log_emiss[0]
        log_alpha -= _logsumexp(log_alpha)
        posteriors[0] = np.exp(log_alpha)

        for t in range(1, n_obs):
            log_alpha_new = np.zeros(n_states)
            for j in range(n_states):
                log_alpha_new[j] = _logsumexp(log_alpha + log_trans[:, j]) + log_emiss[t, j]
            log_alpha_new -= _logsumexp(log_alpha_new)
            posteriors[t] = np.exp(log_alpha_new)
            log_alpha = log_alpha_new

        return posteriors

    def predict_proba_incremental(self, observation: np.ndarray) -> np.ndarray:
        """Process one observation incrementally. Returns [n_states] posterior."""
        assert self._fitted, "Model not fitted"

        log_emiss = self._model._compute_log_likelihood(observation.reshape(1, -1))[0]
        log_trans = np.log(self._model.transmat_ + 1e-300)

        if self._alpha is None:
            log_start = np.log(self._model.startprob_ + 1e-300)
            log_alpha = log_start + log_emiss
        else:
            n_states = self._n_states
            log_alpha = np.zeros(n_states)
            for j in range(n_states):
                log_alpha[j] = _logsumexp(self._alpha + log_trans[:, j]) + log_emiss[j]

        log_alpha -= _logsumexp(log_alpha)
        self._alpha = log_alpha
        return np.exp(log_alpha)

    def reset_filter(self) -> None:
        self._alpha = None

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def transition_matrix(self) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        return self._model.transmat_.copy()

    @property
    def weighted_means(self) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        weights = self._model.weights_
        means = self._model.means_
        return np.einsum("cm,cmf->cf", weights, means)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "n_states": self._n_states,
                "n_mix": self._n_mix,
                "sticky": self._sticky,
                "train_mean": self._train_mean,
                "train_std": self._train_std,
                "fitted": self._fitted,
            }, f)
        logger.info("Saved MicroHMMv2 (n_states=%d) to %s", self._n_states, path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._n_states = data["n_states"]
        self._n_mix = data["n_mix"]
        self._sticky = data["sticky"]
        self._train_mean = data["train_mean"]
        self._train_std = data["train_std"]
        self._fitted = data["fitted"]
        self._alpha = None
        logger.info("Loaded MicroHMMv2 (n_states=%d) from %s", self._n_states, path)
