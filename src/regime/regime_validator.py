"""Regime validation framework for predictability scoring.

Before training regime-specific NNs, this module validates that each
(macro_state, micro_state) pair has enough predictable structure to
justify a dedicated model.

Validation metrics per regime pair:
    - sample_count: must be >= threshold for training
    - baseline_accuracy: logistic regression on simple features
    - mean_magnitude: average |forward return| in ticks
    - directional_consistency: sign(5s ret) == sign(15s ret) rate
    - return_stats: mean, std, skew, kurtosis of forward returns
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MES_TICK = 0.25  # $1.25 per tick, 0.25 price units


@dataclass
class RegimePairMetrics:
    """Validation metrics for a single (macro, micro) regime pair."""
    macro_state: int
    micro_state: int
    n_samples: int
    baseline_accuracy: float  # logistic regression accuracy
    mean_magnitude_ticks: float  # mean |forward return| in ticks
    directional_consistency: float  # sign(5s) == sign(15s) rate
    return_mean: float
    return_std: float
    return_skew: float
    return_kurtosis: float
    classification: str  # "trade", "merge", "skip"
    merge_target: str | None = None  # if merged, which pair


@dataclass
class ValidationReport:
    """Full validation report for all regime pairs."""
    macro_n_states: int
    micro_n_states: int
    pairs: list[RegimePairMetrics]
    tradeable_pairs: list[str]  # list of "macro_micro" keys
    skip_pairs: list[str]
    merge_map: dict[str, str]  # "from" -> "to" pair key


def validate_regime_pairs(
    macro_labels: np.ndarray,
    micro_labels: np.ndarray,
    forward_returns_5s: np.ndarray,
    forward_returns_15s: np.ndarray,
    features: np.ndarray,
    macro_n_states: int,
    micro_n_states: int,
    macro_hmm_means: np.ndarray | None = None,
    micro_hmm_means: np.ndarray | None = None,
    min_samples_trade: int = 1000,
    min_samples_merge: int = 500,
    min_baseline_acc: float = 0.52,
    min_magnitude_ticks: float = 2.7,
) -> ValidationReport:
    """Validate all (macro, micro) regime pairs for NN training worthiness.

    Args:
        macro_labels: [n_windows] macro state labels (0..macro_n_states-1).
        micro_labels: [n_windows] micro state labels (0..micro_n_states-1).
        forward_returns_5s: [n_windows] 5-second forward returns in price units.
        forward_returns_15s: [n_windows] 15-second forward returns in price units.
        features: [n_windows, n_features] simple features for baseline model.
        macro_n_states: number of macro HMM states.
        micro_n_states: number of micro HMM states.
        macro_hmm_means: [macro_n_states, n_macro_features] for merge distance.
        micro_hmm_means: [micro_n_states, n_micro_features] for merge distance.
        min_samples_trade: minimum samples for dedicated model.
        min_samples_merge: minimum samples for merge candidate.
        min_baseline_acc: minimum logistic regression accuracy.
        min_magnitude_ticks: minimum mean |return| in ticks.

    Returns:
        ValidationReport with per-pair metrics and routing map.
    """
    pairs: list[RegimePairMetrics] = []
    tradeable: list[str] = []
    skip_list: list[str] = []
    merge_candidates: list[str] = []

    for m in range(macro_n_states):
        for u in range(micro_n_states):
            mask = (macro_labels == m) & (micro_labels == u)
            n = mask.sum()
            pair_key = f"{m}_{u}"

            if n < 50:
                # Not enough data even to compute stats
                metrics = RegimePairMetrics(
                    macro_state=m, micro_state=u, n_samples=n,
                    baseline_accuracy=0.5, mean_magnitude_ticks=0.0,
                    directional_consistency=0.5,
                    return_mean=0.0, return_std=0.0,
                    return_skew=0.0, return_kurtosis=0.0,
                    classification="skip",
                )
                pairs.append(metrics)
                skip_list.append(pair_key)
                continue

            ret_5 = forward_returns_5s[mask]
            ret_15 = forward_returns_15s[mask]
            feats = features[mask]

            # Forward returns in ticks
            ret_5_ticks = ret_5 / MES_TICK
            ret_15_ticks = ret_15 / MES_TICK

            # Metric 1: Baseline accuracy (logistic regression)
            baseline_acc = _baseline_accuracy(feats, ret_5)

            # Metric 2: Mean magnitude
            mean_mag = float(np.abs(ret_5_ticks).mean())

            # Metric 3: Directional consistency
            dir_5 = np.sign(ret_5)
            dir_15 = np.sign(ret_15)
            # Exclude zeros (no movement)
            nonzero = (dir_5 != 0) & (dir_15 != 0)
            if nonzero.sum() > 10:
                consistency = float((dir_5[nonzero] == dir_15[nonzero]).mean())
            else:
                consistency = 0.5

            # Metric 4: Return distribution stats (pure numpy, no scipy)
            ret_mean = float(ret_5_ticks.mean())
            ret_std = float(ret_5_ticks.std())
            diff = ret_5_ticks - ret_mean
            m2 = float((diff**2).mean())
            m3 = float((diff**3).mean())
            m4 = float((diff**4).mean())
            ret_skew = float(m3 / (m2**1.5)) if m2 > 1e-30 else 0.0
            ret_kurt = float(m4 / (m2**2) - 3.0) if m2 > 1e-30 else 0.0

            # Classification
            if n >= min_samples_trade and baseline_acc > min_baseline_acc and mean_mag > min_magnitude_ticks:
                classification = "trade"
                tradeable.append(pair_key)
            elif n >= min_samples_merge:
                classification = "merge"
                merge_candidates.append(pair_key)
            else:
                classification = "skip"
                skip_list.append(pair_key)

            metrics = RegimePairMetrics(
                macro_state=m, micro_state=u, n_samples=n,
                baseline_accuracy=baseline_acc,
                mean_magnitude_ticks=mean_mag,
                directional_consistency=consistency,
                return_mean=ret_mean, return_std=ret_std,
                return_skew=ret_skew, return_kurtosis=ret_kurt,
                classification=classification,
            )
            pairs.append(metrics)

            logger.info(
                "  Pair (%d,%d): n=%d, baseline=%.3f, mag=%.2f ticks, "
                "consistency=%.3f -> %s",
                m, u, n, baseline_acc, mean_mag, consistency, classification,
            )

    # Resolve merges: find nearest tradeable pair for each merge candidate
    merge_map: dict[str, str] = {}
    if merge_candidates and tradeable and macro_hmm_means is not None and micro_hmm_means is not None:
        merge_map = _resolve_merges(
            merge_candidates, tradeable,
            macro_hmm_means, micro_hmm_means,
            macro_n_states, micro_n_states,
        )
        # Update classification on merged pairs
        for pair in pairs:
            key = f"{pair.macro_state}_{pair.micro_state}"
            if key in merge_map:
                pair.classification = "merge"
                pair.merge_target = merge_map[key]
    elif merge_candidates:
        # No HMM means available — merge into largest tradeable pair
        if tradeable:
            largest = max(tradeable, key=lambda k: next(
                p.n_samples for p in pairs
                if f"{p.macro_state}_{p.micro_state}" == k
            ))
            for mc in merge_candidates:
                merge_map[mc] = largest
        else:
            # No tradeable pairs — all merges become skips
            for mc in merge_candidates:
                skip_list.append(mc)
                for pair in pairs:
                    if f"{pair.macro_state}_{pair.micro_state}" == mc:
                        pair.classification = "skip"

    return ValidationReport(
        macro_n_states=macro_n_states,
        micro_n_states=micro_n_states,
        pairs=pairs,
        tradeable_pairs=tradeable,
        skip_pairs=skip_list,
        merge_map=merge_map,
    )


def _baseline_accuracy(features: np.ndarray, forward_returns: np.ndarray) -> float:
    """Train logistic regression on features to predict forward direction.

    Uses 70/30 chronological split within the regime's data.
    Returns accuracy on the held-out 30%.
    """
    n = len(features)
    if n < 100:
        return 0.5

    labels = (forward_returns > 0).astype(np.int32)

    # Check class balance — if one class < 10%, return 0.5
    pos_rate = labels.mean()
    if pos_rate < 0.1 or pos_rate > 0.9:
        return 0.5

    # Chronological split
    split = int(n * 0.7)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = labels[:split], labels[split:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
    lr.fit(X_train_s, y_train)
    return float(lr.score(X_test_s, y_test))


def _resolve_merges(
    merge_candidates: list[str],
    tradeable: list[str],
    macro_means: np.ndarray,
    micro_means: np.ndarray,
    macro_n_states: int,
    micro_n_states: int,
) -> dict[str, str]:
    """Find nearest tradeable pair for each merge candidate by emission distance."""
    merge_map = {}

    for mc in merge_candidates:
        mc_macro, mc_micro = int(mc.split("_")[0]), int(mc.split("_")[1])
        mc_vec = np.concatenate([macro_means[mc_macro], micro_means[mc_micro]])

        best_dist = float("inf")
        best_target = tradeable[0]

        for t in tradeable:
            t_macro, t_micro = int(t.split("_")[0]), int(t.split("_")[1])
            t_vec = np.concatenate([macro_means[t_macro], micro_means[t_micro]])
            dist = float(np.linalg.norm(mc_vec - t_vec))
            if dist < best_dist:
                best_dist = dist
                best_target = t

        merge_map[mc] = best_target
        logger.info("  Merge (%s) -> (%s), distance=%.3f", mc, best_target, best_dist)

    return merge_map


def save_report(report: ValidationReport, path: str | Path) -> None:
    """Save validation report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "macro_n_states": report.macro_n_states,
        "micro_n_states": report.micro_n_states,
        "pairs": [asdict(p) for p in report.pairs],
        "tradeable_pairs": report.tradeable_pairs,
        "skip_pairs": report.skip_pairs,
        "merge_map": report.merge_map,
    }

    def _convert(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    logger.info("Saved validation report to %s", path)


def load_report(path: str | Path) -> ValidationReport:
    """Load validation report from JSON."""
    with open(path) as f:
        data = json.load(f)

    pairs = [RegimePairMetrics(**p) for p in data["pairs"]]
    return ValidationReport(
        macro_n_states=data["macro_n_states"],
        micro_n_states=data["micro_n_states"],
        pairs=pairs,
        tradeable_pairs=data["tradeable_pairs"],
        skip_pairs=data["skip_pairs"],
        merge_map=data["merge_map"],
    )
