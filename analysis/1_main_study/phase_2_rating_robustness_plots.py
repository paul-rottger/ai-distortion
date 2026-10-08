#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 2 VISUALIZATION: RATING PROTOCOL ROBUSTNESS
#
# Generates figure outputs for the rating-protocol robustness checks.
#
# - Plots writer-vs-model AMEs for scale attributes across robustness specifications.
# - Plots writer-vs-model AMEs by paragraph position (type x position interaction).
# - Plots time spent per paragraph by position.
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
    "position_1": "First paragraph only",
    "positions_1_3": "Paragraphs 1-3",
    "positions_6_10": "Paragraphs 6-10",
    "previous_type": "Control: previous paragraph type",
    "no_speeders": "Excluding speeders",
    "no_straightliners": "Excluding straightliners",
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
    y_offset = get_group_offsets(list(reversed(specifications)), offset_scale=0.1)

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


def plot_ame_by_position(position_df, save_path):
    attributes = [a for a in SCALE_ATTRIBUTES if a in set(position_df["attribute"])]
    n_cols = 5
    n_rows = -(-len(attributes) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows), sharex=True)
    for ax, attribute in zip(axes.flat, attributes):
        df = position_df[position_df["attribute"] == attribute].sort_values("paragraph_position")
        ax.fill_between(df["paragraph_position"], df["ame_low"], df["ame_high"], color="tab:blue", alpha=0.2, linewidth=0)
        ax.plot(df["paragraph_position"], df["ame"], color="tab:blue", marker="o", markersize=3)
        ax.axhline(0, color="black", linestyle=(0, (5, 7)), linewidth=0.8)
        ax.set_title(attribute, fontsize=10)
        ax.set_xticks(range(1, 11))
        sns.despine(ax=ax)
    for ax in list(axes.flat)[len(attributes):]:
        ax.set_visible(False)

    fig.supxlabel("Paragraph position", fontsize=12)
    fig.supylabel("AME of model vs writer paragraph", fontsize=12)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_timing_by_position(timing_df, save_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.fill_between(timing_df["paragraph_position"], timing_df["q25_seconds"], timing_df["q75_seconds"], color="gray", alpha=0.25, linewidth=0, label="Interquartile range")
    ax.plot(timing_df["paragraph_position"], timing_df["median_seconds"], color="black", marker="o", markersize=4, label="Median")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Paragraph position", fontsize=12)
    ax.set_ylabel("Seconds per paragraph", fontsize=12)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_figure(fig, save_path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    plot_ame_by_specification(
        pd.read_csv(RESULTS_DIR / "ame_by_specification.csv"),
        FIGURES_DIR / "ame_by_specification.pdf",
    )
    plot_ame_by_position(
        pd.read_csv(RESULTS_DIR / "ame_by_position.csv"),
        FIGURES_DIR / "ame_by_position.pdf",
    )
    plot_timing_by_position(
        pd.read_csv(RESULTS_DIR / "timing_by_position.csv"),
        FIGURES_DIR / "timing_by_position.pdf",
    )
