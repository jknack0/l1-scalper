"""Regime pair routing for (macro, micro) -> entry model mapping.

Routes live regime state to the appropriate entry model based on
validation results. Handles merges and fallback logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimePairConfig:
    """Configuration for a single (macro, micro) regime pair."""
    macro_state: int
    micro_state: int
    model_id: str           # e.g., "pair_2_1" or "fallback"
    tradeable: bool
    merged_into: str | None  # model_id of merge target, if any
    n_train_samples: int
    baseline_accuracy: float
    mean_magnitude_ticks: float

    @property
    def pair_key(self) -> str:
        return f"{self.macro_state}_{self.micro_state}"

    @property
    def effective_model_id(self) -> str:
        """Model ID to actually load — follows merge chain."""
        return self.merged_into or self.model_id


class RegimePairRouter:
    """Maps (macro_state, micro_state) -> entry model + tradeable flag.

    Built from validation results. Used at inference time to route
    the current regime state to the appropriate entry model.
    """

    def __init__(self, configs: list[RegimePairConfig]) -> None:
        self._lookup: dict[tuple[int, int], RegimePairConfig] = {}
        self._model_ids: set[str] = set()

        for cfg in configs:
            self._lookup[(cfg.macro_state, cfg.micro_state)] = cfg
            if cfg.tradeable:
                self._model_ids.add(cfg.effective_model_id)

        logger.info(
            "RegimePairRouter: %d pairs, %d tradeable, %d unique models",
            len(configs),
            sum(1 for c in configs if c.tradeable),
            len(self._model_ids),
        )

    def route(self, macro_state: int, micro_state: int) -> tuple[str, bool]:
        """Route a regime pair to a model.

        Args:
            macro_state: macro HMM state index.
            micro_state: micro HMM state index.

        Returns:
            (model_id, tradeable) — model_id is "skip" if not tradeable.
        """
        key = (macro_state, micro_state)
        if key not in self._lookup:
            return "fallback", False

        cfg = self._lookup[key]
        if not cfg.tradeable:
            return "skip", False

        return cfg.effective_model_id, True

    @property
    def tradeable_model_ids(self) -> set[str]:
        """Set of unique model IDs that need to be loaded."""
        return self._model_ids.copy()

    @property
    def all_configs(self) -> list[RegimePairConfig]:
        return list(self._lookup.values())

    def save(self, path: str | Path) -> None:
        """Save router config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for cfg in self._lookup.values():
            data.append({
                "macro_state": cfg.macro_state,
                "micro_state": cfg.micro_state,
                "model_id": cfg.model_id,
                "tradeable": cfg.tradeable,
                "merged_into": cfg.merged_into,
                "n_train_samples": cfg.n_train_samples,
                "baseline_accuracy": cfg.baseline_accuracy,
                "mean_magnitude_ticks": cfg.mean_magnitude_ticks,
            })

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_convert)
        logger.info("Saved RegimePairRouter to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> RegimePairRouter:
        """Load router config from JSON."""
        with open(path) as f:
            data = json.load(f)

        configs = [RegimePairConfig(**item) for item in data]
        return cls(configs)


def build_router_from_validation(
    validation_report_path: str | Path,
) -> RegimePairRouter:
    """Build a RegimePairRouter from a validation report JSON.

    Reads the validation report and constructs RegimePairConfig for each pair.
    """
    from src.regime.regime_validator import load_report

    report = load_report(validation_report_path)
    configs: list[RegimePairConfig] = []

    for pair in report.pairs:
        pair_key = f"{pair.macro_state}_{pair.micro_state}"
        model_id = f"pair_{pair_key}"

        if pair.classification == "trade":
            tradeable = True
            merged_into = None
        elif pair.classification == "merge" and pair_key in report.merge_map:
            tradeable = True
            target_key = report.merge_map[pair_key]
            merged_into = f"pair_{target_key}"
        else:
            tradeable = False
            merged_into = None

        configs.append(RegimePairConfig(
            macro_state=pair.macro_state,
            micro_state=pair.micro_state,
            model_id=model_id,
            tradeable=tradeable,
            merged_into=merged_into,
            n_train_samples=pair.n_samples,
            baseline_accuracy=pair.baseline_accuracy,
            mean_magnitude_ticks=pair.mean_magnitude_ticks,
        ))

    return RegimePairRouter(configs)
