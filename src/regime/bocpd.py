"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Detects regime transitions in real-time by maintaining a distribution
over possible run lengths (time since last changepoint).

Uses a normal-inverse-gamma conjugate prior as the observation model.
"""

from __future__ import annotations

import numpy as np

# Default hazard rate: expect a regime change every ~250 bars (≈4 hours at 1-min bars)
DEFAULT_HAZARD_RATE = 1.0 / 250.0


class NormalInverseGamma:
    """Sufficient statistics for the normal-inverse-gamma conjugate prior.

    Tracks per-run-length sufficient statistics for online Bayesian updating.
    """

    def __init__(self, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0) -> None:
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def log_predictive(self, x: float, mu: float, kappa: float,
                       alpha: float, beta: float) -> float:
        """Log predictive probability of observation x under the Student-t posterior."""
        from math import lgamma, log, pi

        nu = 2.0 * alpha
        var = beta * (kappa + 1.0) / (alpha * kappa)

        if var <= 0:
            var = 1e-10

        # Student-t log PDF
        log_p = (
            lgamma((nu + 1.0) / 2.0)
            - lgamma(nu / 2.0)
            - 0.5 * log(nu * pi * var)
            - ((nu + 1.0) / 2.0) * log(1.0 + (x - mu) ** 2 / (nu * var))
        )
        return log_p


class BOCPD:
    """Bayesian Online Changepoint Detection.

    Processes one scalar observation at a time and returns the
    posterior probability that a changepoint occurred.
    """

    def __init__(
        self,
        hazard_rate: float = DEFAULT_HAZARD_RATE,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 500,
    ) -> None:
        self._hazard = hazard_rate
        self._log_hazard = np.log(hazard_rate)
        self._log_1m_hazard = np.log(1.0 - hazard_rate)
        self._max_rl = max_run_length
        self._nig = NormalInverseGamma(mu0, kappa0, alpha0, beta0)

        # Run length distribution: log P(r_t | x_{1:t})
        # Index i = run length i
        self._log_R = np.array([0.0])  # start with run length 0, prob 1

        # Sufficient statistics per run length
        self._mu = np.array([mu0])
        self._kappa = np.array([kappa0])
        self._alpha = np.array([alpha0])
        self._beta = np.array([beta0])

    def detect(self, observation: float) -> float:
        """Process one observation and return changepoint probability.

        The observation should be a scalar summary of the current bar
        (e.g., realized volatility, or a composite feature).

        Returns:
            Probability of a changepoint at the current time step [0, 1].
        """
        x = observation
        n = len(self._log_R)

        # 1. Evaluate predictive probabilities for each run length
        log_pred = np.array([
            self._nig.log_predictive(x, self._mu[i], self._kappa[i],
                                     self._alpha[i], self._beta[i])
            for i in range(n)
        ])

        # 2. Growth probabilities: P(r_t = r_{t-1} + 1, x_{1:t})
        log_growth = self._log_R + log_pred + self._log_1m_hazard

        # 3. Changepoint probability: P(r_t = 0, x_{1:t})
        log_cp = _logsumexp(self._log_R + log_pred + self._log_hazard)

        # 4. New run length distribution
        new_log_R = np.empty(n + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize
        log_evidence = _logsumexp(new_log_R)
        new_log_R -= log_evidence

        # 5. Update sufficient statistics
        new_mu = np.empty(n + 1)
        new_kappa = np.empty(n + 1)
        new_alpha = np.empty(n + 1)
        new_beta = np.empty(n + 1)

        # Run length 0: reset to prior
        new_mu[0] = self._nig.mu0
        new_kappa[0] = self._nig.kappa0
        new_alpha[0] = self._nig.alpha0
        new_beta[0] = self._nig.beta0

        # Run lengths 1..n: update from previous
        new_kappa[1:] = self._kappa + 1.0
        new_mu[1:] = (self._kappa * self._mu + x) / new_kappa[1:]
        new_alpha[1:] = self._alpha + 0.5
        new_beta[1:] = (
            self._beta
            + 0.5 * self._kappa * (x - self._mu) ** 2 / new_kappa[1:]
        )

        # Truncate to max run length
        if len(new_log_R) > self._max_rl:
            new_log_R = new_log_R[: self._max_rl]
            new_mu = new_mu[: self._max_rl]
            new_kappa = new_kappa[: self._max_rl]
            new_alpha = new_alpha[: self._max_rl]
            new_beta = new_beta[: self._max_rl]
            # Re-normalize after truncation
            log_evidence = _logsumexp(new_log_R)
            new_log_R -= log_evidence

        self._log_R = new_log_R
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        # Changepoint probability = P(r_t = 0)
        return float(np.exp(new_log_R[0]))

    def reset(self) -> None:
        """Reset to initial state."""
        self._log_R = np.array([0.0])
        self._mu = np.array([self._nig.mu0])
        self._kappa = np.array([self._nig.kappa0])
        self._alpha = np.array([self._nig.alpha0])
        self._beta = np.array([self._nig.beta0])


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    m = log_x.max()
    return float(m + np.log(np.sum(np.exp(log_x - m))))
