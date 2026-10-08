#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 2 VISUALIZATION: WRITER ENGAGEMENT ROBUSTNESS
#
# Generates figure outputs for the writer-engagement robustness checks.
#
# - Plots writer-vs-model AMEs for scale attributes on the full sample and after
#   excluding writer-proposition pairs with low self-reported issue knowledge,
#   importance, or confidence.
# - Saves plots to `figures/main_phase_2_robustness/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import os
import sys
from pathlib import Path

import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "utils_py"))
REPO_ROOT = Path(BASE_DIR).resolve().parents[1]

# Internal imports
sys.path.insert(0, UTILS_DIR)
from demo_paths import get_figures_dir, get_results_input_dir, parse_demo_mode
from variable_definitions import SCALE_ATTRIBUTES
from plotting_utils import get_group_offsets

DEMO_MODE = parse_demo_mode()
RESULTS_DIR = get_results_input_dir(REPO_ROOT, "main_phase_2_robustness", demo_mode=DEMO_MODE)
FIGURES_DIR = get_figures_dir(REPO_ROOT, "main_phase_2_robustness", demo_mode=DEMO_MODE)

# Plot configuration
SPECIFICATION_LABELS = {
    "full": "Full sample",
    "no_low_knowledge": "Excluding low issue knowledge (<50)",
    "no_low_importance": "Excluding low issue importance (<50)",
    "no_low_confidence": "Excluding low confidence in opinion (<50)",
}
SPECIFICATION_COLORS = dict(zip(SPECIFICATION_LABELS, ["black"] + sns.color_palette("tab10", len(SPECIFICATION_LABELS) - 1)))


def save_figure(fig, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_ame_by_specification(ame_df, save_path):
    attributes = [a for a in SCALE_ATTRIBUTES if a in set(ame_df["attribute"])]
    attribute_order = list(reversed(attributes))
    attribute_positions = {attribute: index for index, attribute in enumerate(attribute_order)}
    specifications = [s for s in SPECIFICATION_LABELS if s in set(ame_df["specification"])]
    y_offset = get_group_offsets(list(reversed(specifications)), offset_scale=0.15)

    fig, ax = plt.subplots(figsize=(8, 14))
    for specification in specifications:
        df = ame_df[ame_df["specification"] == specification]
        ax.errorbar(
            df["ame"],
            [attribute_positions[a] + y_offset[specification] for a in df["attribute"]],
            xerr=[df["ame"] - df["ame_low"], df["ame_high"] - df["ame"]],
            fmt="o",
            markersize=3,
            capsize=0,
            elinewidth=1,
            color=SPECIFICATION_COLORS[specification],
            label=SPECIFICATION_LABELS[specification],
            zorder=4,
        )

    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)
    for boundary in range(len(attribute_order) - 1):
        ax.axhline(boundary + 0.5, color="gray", linestyle=(0, (5, 5)), linewidth=0.3)
    ax.set_yticks(range(len(attribute_order)))
    ax.set_yticklabels(attribute_order)
    ax.set_ylim(-0.6, len(attribute_order) - 0.4)
    ax.set_xlabel("AME of model vs writer paragraph (0-100 scale)", fontsize=12)
    ax.set_ylabel("Attribute", fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, save_path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    plot_ame_by_specification(
        pd.read_csv(RESULTS_DIR / "engagement_ame_by_specification.csv"),
        FIGURES_DIR / "engagement_ame_by_specification.pdf",
    )
