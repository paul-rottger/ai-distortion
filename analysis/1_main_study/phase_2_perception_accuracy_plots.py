#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 2 VISUALIZATION: PERCEPTION ACCURACY
#
# Generates figure outputs for the comparison of reader perceptions with
# writer self-reports.
#
# - Plots correlation (tracking) and mean signed error (bias) between perceived
#   and self-reported scale attributes, for writer and model paragraphs.
# - Plots Cohen's kappa and ordinal correlations for demographic attributes.
# - Plots model - writer differences by model input condition.
# - Plots row-normalised confusion matrices for selected ordinal demographics.
# - Saves plots to `figures/main_phase_2_accuracy/`.
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
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "utils_py"))
REPO_ROOT = Path(BASE_DIR).resolve().parents[1]

# Internal imports
sys.path.insert(0, UTILS_DIR)
from demo_paths import get_figures_dir, get_results_input_dir, parse_demo_mode
from plotting_utils import get_group_offsets

DEMO_MODE = parse_demo_mode()
RESULTS_DIR = get_results_input_dir(REPO_ROOT, "main_phase_2_accuracy", demo_mode=DEMO_MODE)
FIGURES_DIR = get_figures_dir(REPO_ROOT, "main_phase_2_accuracy", demo_mode=DEMO_MODE)

# Plot configuration
PARAGRAPH_TYPE_COLORS = {"writer": "#0070C0", "model": "#7030A0"}
PARAGRAPH_TYPE_LABELS = {"writer": "Writer paragraph", "model": "AI paragraph"}
SPLIT_LABELS = {"unedited": "Unedited", "edited": "Edited", "preferred": "Preferred"}
LEVEL_LABELS = {"rater": "Individual raters", "paragraph": "Aggregated across raters"}
INPUT_CONDITION_LABELS = {
	"stance-based": "Stance-based",
	"bullets-based": "Bullets-based",
	"rewrite": "Rewrite",
	"improve": "Improve",
}
INPUT_CONDITION_COLORS = dict(zip(INPUT_CONDITION_LABELS, sns.color_palette("tab10", len(INPUT_CONDITION_LABELS))))
SCALE_LABELS = {
	"writer_stance": "Stance",
	"writer_stance_polarity": "Stance polarity",
	"writer_knowledge": "Knowledge",
	"writer_importance": "Importance",
	"writer_confidence": "Confidence",
	"writer_affect_x": "Affect: valence",
	"writer_affect_y": "Affect: arousal",
}
CATEGORICAL_LABELS = {
	"writer_age_binned": "Age group",
	"writer_education": "Education",
	"writer_income": "Income",
	"writer_english_skills": "English skills",
	"writer_politicalIdeology": "Political ideology",
	"writer_english_first": "English first language",
	"writer_gender": "Gender",
	"writer_race": "Ethnicity",
	"writer_politicalParty": "Political party",
}
METRIC_LABELS = {
	"correlation": "Correlation with self-report (r)",
	"bias": "Mean signed error (perceived - self-report)",
	"kappa": "Cohen's kappa",
	"ordinal_correlation": "Correlation of ordinal codes (r)",
}
CONFUSION_ATTRIBUTES = ["writer_age_binned", "writer_education", "writer_english_skills", "writer_politicalIdeology"]
CONFUSION_SPLIT = "preferred"


def save_figure(fig, save_path):
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved: {save_path}")


# =============================================================================
# PLOTS
# =============================================================================

def draw_dot_whiskers(ax, df, attribute_labels, groups, group_column, colors, labels):
	"""Horizontal dot-whisker plot with one row per attribute and offset groups."""
	attribute_order = [a for a in reversed(attribute_labels) if a in set(df["attribute"])]
	attribute_positions = {attribute: index for index, attribute in enumerate(attribute_order)}
	y_offset = get_group_offsets(list(reversed(groups)), offset_scale=0.22 if len(groups) == 2 else 0.15)

	for group in groups:
		group_df = df[(df[group_column] == group) & df["attribute"].isin(attribute_positions)]
		ax.errorbar(
			group_df["estimate"],
			[attribute_positions[a] + y_offset[group] for a in group_df["attribute"]],
			xerr=[group_df["estimate"] - group_df["ci_low"], group_df["ci_high"] - group_df["estimate"]],
			fmt="o",
			markersize=4,
			capsize=0,
			elinewidth=1.2,
			color=colors[group],
			label=labels[group],
			zorder=4,
		)

	ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)
	for boundary in range(len(attribute_order) - 1):
		ax.axhline(boundary + 0.5, color="gray", linestyle=(0, (5, 5)), linewidth=0.3)
	ax.set_yticks(range(len(attribute_order)))
	ax.set_yticklabels([attribute_labels[a] for a in attribute_order])
	ax.set_ylim(-0.6, len(attribute_order) - 0.4)
	sns.despine(ax=ax)


def plot_by_split(df, metric, attribute_labels, row_column, row_labels, save_path, panel_height):
	"""Grid of dot-whisker panels: rows by `row_column`, columns by split."""
	df = df[df["metric"] == metric] if metric else df
	rows = [r for r in row_labels if r in set(df[row_column])]
	splits = [s for s in SPLIT_LABELS if s in set(df["split"])]

	fig, axes = plt.subplots(len(rows), len(splits), figsize=(4 * len(splits), panel_height * len(rows)), sharey="row", squeeze=False)
	for row_index, row in enumerate(rows):
		for col_index, split in enumerate(splits):
			ax = axes[row_index, col_index]
			panel_df = df[(df[row_column] == row) & (df["split"] == split)]
			draw_dot_whiskers(ax, panel_df, attribute_labels, list(PARAGRAPH_TYPE_COLORS), "paragraph_type", PARAGRAPH_TYPE_COLORS, PARAGRAPH_TYPE_LABELS)
			if row_index == 0:
				ax.set_title(f"{SPLIT_LABELS[split]} split", fontsize=12)
			if col_index == 0:
				ax.set_ylabel(row_labels[row], fontsize=12)
			ax.set_xlabel(METRIC_LABELS[metric or row], fontsize=10)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
	plt.tight_layout()
	save_figure(fig, save_path)


def plot_by_input_condition(condition_df, save_path):
	differences = condition_df[condition_df["paragraph_type"] == "model - writer"]
	panels = [
		("scale", "correlation", SCALE_LABELS),
		("categorical", "ordinal_correlation", CATEGORICAL_LABELS),
		("categorical", "kappa", CATEGORICAL_LABELS),
	]
	conditions = [c for c in INPUT_CONDITION_LABELS if c in set(differences["input_condition"])]

	fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.5))
	for ax, (kind, metric, attribute_labels) in zip(axes, panels):
		panel_df = differences[(differences["kind"] == kind) & (differences["metric"] == metric)]
		draw_dot_whiskers(ax, panel_df, attribute_labels, conditions, "input_condition", INPUT_CONDITION_COLORS, INPUT_CONDITION_LABELS)
		ax.xaxis.set_major_locator(MaxNLocator(5))
		ax.set_xlabel(f"Model - writer difference\n{METRIC_LABELS[metric]}", fontsize=10)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, title="Model input condition", frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
	plt.tight_layout()
	save_figure(fig, save_path)


def plot_confusion_matrices(confusion_df, save_path):
	df = confusion_df[(confusion_df["split"] == CONFUSION_SPLIT) & confusion_df["attribute"].isin(CONFUSION_ATTRIBUTES)]
	paragraph_types = list(PARAGRAPH_TYPE_LABELS)

	fig, axes = plt.subplots(len(CONFUSION_ATTRIBUTES), len(paragraph_types), figsize=(11, 4.2 * len(CONFUSION_ATTRIBUTES)), layout="constrained")
	for row_index, attribute in enumerate(CONFUSION_ATTRIBUTES):
		attribute_df = df[df["attribute"] == attribute]
		levels = list(dict.fromkeys(attribute_df["self_reported"]))
		vmax = attribute_df["row_share"].max()
		for col_index, paragraph_type in enumerate(paragraph_types):
			ax = axes[row_index, col_index]
			matrix = (
				attribute_df[attribute_df["paragraph_type"] == paragraph_type]
				.pivot(index="self_reported", columns="perceived", values="row_share")
				.reindex(index=levels, columns=levels)
			)
			sns.heatmap(
				matrix,
				ax=ax,
				cmap="Blues",
				vmin=0,
				vmax=vmax,
				annot=matrix.map(lambda v: "" if np.isnan(v) else f"{v:.0%}"),
				fmt="",
				annot_kws={"fontsize": 7},
				cbar=col_index == len(paragraph_types) - 1,
				cbar_kws={"label": "Share of self-reported row"},
				linewidths=0.5,
				linecolor="white",
			)
			ax.set_title(f"{CATEGORICAL_LABELS[attribute]}: {PARAGRAPH_TYPE_LABELS[paragraph_type]}", fontsize=11)
			ax.set_xlabel("Perceived by reader", fontsize=10)
			ax.set_ylabel("Self-reported by writer" if col_index == 0 else "", fontsize=10)
			ax.tick_params(axis="both", labelsize=8)
			ax.tick_params(axis="y", labelrotation=0)
			if col_index > 0:
				ax.set_yticklabels([])

	save_figure(fig, save_path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
	scale_df = pd.read_csv(RESULTS_DIR / "scale_accuracy.csv")
	categorical_df = pd.read_csv(RESULTS_DIR / "categorical_accuracy.csv")

	plot_by_split(scale_df, "correlation", SCALE_LABELS, "level", LEVEL_LABELS, FIGURES_DIR / "scale_tracking.pdf", panel_height=4)
	plot_by_split(scale_df, "bias", SCALE_LABELS, "level", LEVEL_LABELS, FIGURES_DIR / "scale_bias.pdf", panel_height=4)
	plot_by_split(
		categorical_df[(categorical_df["level"] == "rater") & categorical_df["metric"].isin(["kappa", "ordinal_correlation"])],
		None,
		CATEGORICAL_LABELS,
		"metric",
		{"kappa": "Cohen's kappa", "ordinal_correlation": "Ordinal correlation"},
		FIGURES_DIR / "categorical_kappa.pdf",
		panel_height=4.5,
	)
	plot_by_input_condition(pd.read_csv(RESULTS_DIR / "by_input_condition.csv"), FIGURES_DIR / "by_input_condition.pdf")
	plot_confusion_matrices(pd.read_csv(RESULTS_DIR / "confusion_matrices.csv"), FIGURES_DIR / "confusion_matrices.pdf")
