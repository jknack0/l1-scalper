"""Regime visualization: price with regime overlays and BOCPD changepoints."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Regime colors
REGIME_COLORS = {
    0: "#2ecc71",  # TRENDING = green
    1: "#3498db",  # MEAN_REVERTING = blue
    2: "#e74c3c",  # CHOPPY = red
}

REGIME_LABELS = {
    0: "Trending",
    1: "Mean-Reverting",
    2: "Choppy",
}


def plot_regime_overlay(
    prices: np.ndarray,
    regime_labels: np.ndarray,
    changepoint_probs: np.ndarray | None = None,
    posteriors: np.ndarray | None = None,
    title: str = "Market Regimes",
    output_path: str | Path | None = None,
) -> None:
    """Plot price chart with background colored by regime.

    Args:
        prices: shape [n] — price series (e.g., mid prices per minute)
        regime_labels: shape [n] — integer regime labels (0, 1, 2)
        changepoint_probs: shape [n] — BOCPD changepoint probabilities
        posteriors: shape [n, 3] — regime posterior probabilities
        title: chart title
        output_path: save as PNG if provided, else show
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_panels = 1
    if posteriors is not None:
        n_panels += 1
    if changepoint_probs is not None:
        n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(prices))

    # Panel 1: Price with regime background
    ax = axes[0]
    ax.plot(x, prices, color="black", linewidth=0.8, alpha=0.9)

    # Color background by regime
    _fill_regime_background(ax, regime_labels, prices.min(), prices.max())

    # Mark changepoints
    if changepoint_probs is not None:
        cp_mask = changepoint_probs > 0.7
        for idx in np.where(cp_mask)[0]:
            ax.axvline(x=idx, color="purple", linewidth=1.5, alpha=0.7, linestyle="--")

    legend_elements = [Patch(facecolor=c, label=REGIME_LABELS[k], alpha=0.3)
                       for k, c in REGIME_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_ylabel("Price")
    ax.set_title(title)

    panel_idx = 1

    # Panel 2: Posterior probabilities (stacked area)
    if posteriors is not None:
        ax = axes[panel_idx]
        ax.stackplot(
            x,
            posteriors[:, 0],
            posteriors[:, 1],
            posteriors[:, 2],
            labels=["Trending", "Mean-Reverting", "Choppy"],
            colors=[REGIME_COLORS[0], REGIME_COLORS[1], REGIME_COLORS[2]],
            alpha=0.7,
        )
        ax.set_ylabel("Posterior Probability")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        panel_idx += 1

    # Panel 3: Changepoint probability
    if changepoint_probs is not None:
        ax = axes[panel_idx]
        ax.plot(x, changepoint_probs, color="purple", linewidth=0.8)
        ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Threshold (0.7)")
        ax.set_ylabel("Changepoint Prob")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Bar Index (1-minute)")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved regime plot to %s", output_path)
        plt.close(fig)
    else:
        plt.show()


def _fill_regime_background(ax, labels: np.ndarray, ymin: float, ymax: float) -> None:
    """Fill background spans with regime colors."""
    if len(labels) == 0:
        return

    current_regime = labels[0]
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_regime:
            color = REGIME_COLORS.get(current_regime, "#cccccc")
            ax.axvspan(start, i, alpha=0.15, color=color)
            current_regime = labels[i]
            start = i

    # Last segment
    color = REGIME_COLORS.get(current_regime, "#cccccc")
    ax.axvspan(start, len(labels), alpha=0.15, color=color)
