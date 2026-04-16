#!/usr/bin/env python3


# =============================================================================
# SETUP
# =============================================================================

# Package imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils_py"))
RESULTS_DIR = os.path.normpath(
	os.path.join(BASE_DIR, "../../results/followup_mitigation_phase_2_distortion")
)
CORRELATION_RESULTS_DIR = os.path.normpath(
	os.path.join(BASE_DIR, "../../results/followup_mitigation_phase_2_distribution")
)
FIGURES_DIR = os.path.normpath(
	os.path.join(BASE_DIR, "../../figures/followup_mitigation_phase_2_distortion")
)
DISTORTION_TOLERANCE_PATH = os.path.normpath(
	os.path.join(BASE_DIR, "../../data/main_phase_1/distortion_responses_summary.csv")
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PREFERRED_PARA_TYPE = "preferred"
MITIGATIONS = ["prompting", "reranking"]
CHANGE_VS_STANCE_MITIGATIONS = ["reranking"]
CORRELATION_TARGET_ATTRIBUTE = "writer_stance_polarity"
EXCLUDED_ATTRIBUTES = {"writer_affect_x", "writer_affect_y"}
CHANGE_VS_STANCE_BASE_MARKER_SIZE = 6
CHANGE_VS_STANCE_MAX_MARKER_SCALE = 3
REDUCTION_ARROW_COLOR = "#2f6f62"
BEST_FIT_LINE_COLOR = "#4c5c68"
CHANGE_GROUP_OVERRIDES = {
	"writer_openness": "liked",
}

DISTORTION_TOLERANCE_BAND_COLORS = {
	"0_25": "#cc0000",
	"25_50": "#e06666",
	"50_75": "#93c47d",
	"75_100": "#38761d",
}
ARROW_COLORS = {
	"red": "#cc0000",
	"green": "#38761d",
}
OUTCOME_MAP = {
	"writer_stance_polarity": ("extreme_more", "moderate_more"),
	"writer_confidence": ("confidence_more", "confidence_less"),
	"writer_knowledge": ("knowledge_more", "knowledge_less"),
	"writer_importance": ("importance_more", "importance_less"),
	"paragraph_relevance": ("relevance_more", "relevance_less"),
	"paragraph_clarity": ("clarity_more", "clarity_less"),
	"paragraph_formality": ("formality_more", "formality_less"),
	"paragraph_informativeness": ("informative_more", "informative_less"),
	"paragraph_originality": ("originality_more", "originality_less"),
	"writer_friendliness": ("friendliness_more", "friendliness_less"),
	"writer_optimism": ("optimism_more", "optimism_less"),
	"writer_community": ("community_more", "community_less"),
	"writer_openness": ("open_views_more", "open_views_less"),
	"paragraph_hope": ("hope_more", "hope_less"),
	"paragraph_excitement": ("excitement_more", "excitement_less"),
	"paragraph_fear": ("fear_more", "fear_less"),
	"paragraph_disgust": ("disgust_more", "disgust_less"),
	"paragraph_anger": ("anger_more", "anger_less"),
}

SIGNIFICANCE_DETAILS_PATH = os.path.join(
	RESULTS_DIR,
	PREFERRED_PARA_TYPE,
	"mitigation_significance_details.csv",
)
CORRELATION_RESULTS_PATH = os.path.join(
	CORRELATION_RESULTS_DIR,
	PREFERRED_PARA_TYPE,
	"scale_attribute_correlations.csv",
)

DISTORTION_TOLERANCE_LOOKUP: dict[str, float] = {}


# =============================================================================
# LOAD DATA
# =============================================================================


def load_significance_details(significance_details_path=SIGNIFICANCE_DETAILS_PATH):
	if not os.path.exists(significance_details_path):
		return pd.DataFrame()
	return pd.read_csv(significance_details_path)


def load_correlation_data(correlation_results_path=CORRELATION_RESULTS_PATH):
	if not os.path.exists(correlation_results_path):
		return pd.DataFrame()
	return pd.read_csv(correlation_results_path)


def load_distortion_tolerance_lookup(
	distortion_tolerance_path=DISTORTION_TOLERANCE_PATH,
):
	distortion_tolerance_df = pd.read_csv(distortion_tolerance_path)
	return distortion_tolerance_df.set_index("distortion")["mean"].to_dict()


# =============================================================================
# ANALYSIS
# =============================================================================


def format_p_value(p_value):
	if pd.isna(p_value):
		return "NA"
	if p_value < 0.001:
		return "< .001"
	return f"= {p_value:.3f}"


def get_significance_stars(p_value):
	if pd.isna(p_value):
		return ""
	if p_value < 0.001:
		return "***"
	if p_value < 0.01:
		return "**"
	if p_value < 0.05:
		return "*"
	return ""


def get_plot_correlations(plot_df, x_column):
	valid_df = plot_df[[x_column, "stance_correlation"]].dropna()

	if len(valid_df) < 3:
		return None

	pearson_r, pearson_p = pearsonr(
		valid_df[x_column],
		valid_df["stance_correlation"],
	)
	spearman_r, spearman_p = spearmanr(
		valid_df[x_column],
		valid_df["stance_correlation"],
	)

	return {
		"pearson_r": pearson_r,
		"pearson_p": pearson_p,
		"spearman_r": spearman_r,
		"spearman_p": spearman_p,
	}


def get_pairwise_correlation(x_values, y_values):
	pair_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

	if len(pair_df) < 3:
		return None

	if pair_df["x"].nunique() < 2 or pair_df["y"].nunique() < 2:
		return None

	pearson_r, pearson_p = pearsonr(pair_df["x"], pair_df["y"])
	spearman_r, spearman_p = spearmanr(pair_df["x"], pair_df["y"])

	return {
		"pearson_r": pearson_r,
		"pearson_p": pearson_p,
		"spearman_r": spearman_r,
		"spearman_p": spearman_p,
	}


def get_linear_fit_points(x_values, y_values, num_points=200):
	pair_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

	if len(pair_df) < 2 or pair_df["x"].nunique() < 2:
		return None

	slope, intercept = np.polyfit(pair_df["x"], pair_df["y"], 1)
	x_fit = np.linspace(pair_df["x"].min(), pair_df["x"].max(), num_points)
	y_fit = slope * x_fit + intercept

	return {
		"slope": float(slope),
		"intercept": float(intercept),
		"x_fit": x_fit,
		"y_fit": y_fit,
	}


def get_fraction_toward_writer_ame(ame_value, writer_ame_value):
	if pd.isna(ame_value) or pd.isna(writer_ame_value) or writer_ame_value == 0:
		return 0.0

	if np.sign(ame_value) != np.sign(writer_ame_value):
		return 0.0

	return float(np.clip(abs(ame_value) / abs(writer_ame_value), 0.0, 1.0))


def get_change_vs_stance_marker_size(
	is_significant,
	ame_value,
	writer_ame_value,
	base_marker_size=CHANGE_VS_STANCE_BASE_MARKER_SIZE,
	max_marker_scale=CHANGE_VS_STANCE_MAX_MARKER_SCALE,
):
	if not is_significant:
		return base_marker_size

	fraction_toward_writer = get_fraction_toward_writer_ame(ame_value, writer_ame_value)
	return base_marker_size * (
		1 + fraction_toward_writer * (max_marker_scale - 1)
	)


def get_distortion_label(attribute, ame_value):
	labels = OUTCOME_MAP.get(attribute)
	if labels is None or pd.isna(ame_value) or ame_value == 0:
		return None
	return labels[1] if ame_value > 0 else labels[0]


def get_tolerance_band_color(tolerance_mean):
	if pd.isna(tolerance_mean):
		return None
	if tolerance_mean < 25:
		return DISTORTION_TOLERANCE_BAND_COLORS["0_25"]
	if tolerance_mean < 50:
		return DISTORTION_TOLERANCE_BAND_COLORS["25_50"]
	if tolerance_mean < 75:
		return DISTORTION_TOLERANCE_BAND_COLORS["50_75"]
	return DISTORTION_TOLERANCE_BAND_COLORS["75_100"]


def get_change_group(attribute, background_color, effect_value):
	override_group = CHANGE_GROUP_OVERRIDES.get(attribute)
	if override_group is not None:
		return override_group

	if pd.isna(effect_value) or background_color is None:
		return None

	if background_color in {
		DISTORTION_TOLERANCE_BAND_COLORS["0_25"],
		DISTORTION_TOLERANCE_BAND_COLORS["25_50"],
	}:
		if effect_value < 0:
			return "liked"
		if effect_value > 0:
			return "disliked"
		return None

	if background_color in {
		DISTORTION_TOLERANCE_BAND_COLORS["50_75"],
		DISTORTION_TOLERANCE_BAND_COLORS["75_100"],
	}:
		if effect_value > 0:
			return "liked"
		if effect_value < 0:
			return "disliked"
		return None

	return None

def build_side_effect_correlation_rows(
	significance_details_df,
	correlation_df,
	mitigation,
	correlation_target=CORRELATION_TARGET_ATTRIBUTE,
	significant_only=True,
):
	if significance_details_df.empty or correlation_df.empty:
		return pd.DataFrame()

	mitigation_df = significance_details_df[
		significance_details_df["mitigation"] == mitigation
	].copy()
	mitigation_df = mitigation_df[
		~mitigation_df["attribute"].isin(EXCLUDED_ATTRIBUTES)
	]

	if significant_only:
		mitigation_df = mitigation_df[mitigation_df["significant"]]

	if mitigation_df.empty:
		return pd.DataFrame()

	stance_correlation_df = correlation_df[
		correlation_df["attribute_y"] == correlation_target
	][["attribute_x", "correlation", "p_value"]].rename(
		columns={
			"attribute_x": "attribute",
			"correlation": "stance_correlation",
			"p_value": "correlation_p_value",
		}
	)

	plot_df = mitigation_df.merge(stance_correlation_df, on="attribute", how="inner")

	if plot_df.empty:
		return pd.DataFrame()

	plot_df["attribute_label"] = plot_df["attribute"].str.replace("_", " ")

	plot_df["background_color"] = plot_df.apply(
		lambda row: get_tolerance_band_color(
			DISTORTION_TOLERANCE_LOOKUP.get(
				get_distortion_label(row["attribute"], row["writer_ame"])
			)
		),
		axis=1,
	)
	plot_df["change_group"] = plot_df.apply(
		lambda row: get_change_group(
			row["attribute"],
			row["background_color"],
			row["ame"],
		),
		axis=1,
	)

	return plot_df.dropna(
		subset=[
			"ame",
			"stance_correlation",
		]
	)


def build_mitigation_reduction_rows(significance_details_df, mitigation):
	if significance_details_df.empty:
		return pd.DataFrame()

	plot_df = significance_details_df[
		significance_details_df["mitigation"] == mitigation
	].copy()
	plot_df = plot_df[~plot_df["attribute"].isin(EXCLUDED_ATTRIBUTES)].copy()
	plot_df = plot_df[plot_df["significant"]].copy()

	if plot_df.empty:
		return pd.DataFrame()

	plot_df["reduction_fraction"] = plot_df.apply(
		lambda row: get_fraction_toward_writer_ame(row["ame"], row["writer_ame"]),
		axis=1,
	)
	plot_df = plot_df[plot_df["reduction_fraction"] > 0].copy()

	if plot_df.empty:
		return pd.DataFrame()

	plot_df["background_color"] = plot_df.apply(
		lambda row: get_tolerance_band_color(
			DISTORTION_TOLERANCE_LOOKUP.get(
				get_distortion_label(row["attribute"], row["writer_ame"])
			)
		),
		axis=1,
	)
	plot_df["change_group"] = plot_df.apply(
		lambda row: get_change_group(
			row["attribute"],
			row["background_color"],
			row["ame"],
		),
		axis=1,
	)
	plot_df = plot_df[plot_df["change_group"].isin(["liked", "disliked"])].copy()

	if plot_df.empty:
		return pd.DataFrame()

	plot_df["reduction_percent"] = -plot_df["reduction_fraction"] * 100
	plot_df["attribute_label"] = plot_df.apply(
		lambda row: row["attribute"].replace("_", " ") + get_significance_stars(row["p_value"]),
		axis=1,
	)
	plot_df = plot_df.sort_values(
		by=["reduction_fraction", "attribute_label"],
		ascending=[False, True],
	).reset_index(drop=True)

	return plot_df


def create_mitigation_reduction_plot(
	significance_details_df,
	mitigation,
	xlabel="Reduction in distortion toward writer (%)",
	save_path=None,
):
	plot_df = build_mitigation_reduction_rows(significance_details_df, mitigation)

	if plot_df.empty:
		return None, None

	fig_height = max(4.0, 0.38 * len(plot_df) + 1.2)
	fig, ax = plt.subplots(figsize=(9, fig_height))
	y_positions = np.arange(len(plot_df))
	arrow_colors = {
		"liked": ARROW_COLORS["green"],
		"disliked": ARROW_COLORS["red"],
	}

	for y_position, row in zip(y_positions, plot_df.itertuples(index=False)):
		point_color = arrow_colors[row.change_group]
		ax.annotate(
			"",
			xy=(row.reduction_percent, y_position),
			xytext=(0, y_position),
			arrowprops={
				"arrowstyle": "-|>",
				"color": point_color,
				"lw": 1.8,
				"mutation_scale": 10,
				"shrinkA": 0,
				"shrinkB": 0,
			},
			annotation_clip=False,
		)
		ax.plot(
			row.reduction_percent,
			y_position,
			linestyle="",
			marker="o",
			markersize=4.5,
			color=point_color,
			markeredgewidth=0,
			zorder=3,
		)
		ax.text(
			row.reduction_percent - 1.0,
			y_position,
			f"{row.reduction_percent:.1f}%",
			va="center",
			ha="right",
			fontsize=9,
		)

	ax.set_yticks(y_positions)
	ax.set_yticklabels(plot_df["attribute_label"])
	ax.yaxis.tick_right()
	ax.tick_params(axis="y", labelright=True, labelleft=False)
	ax.invert_yaxis()
	ax.set_xlim(min(-100, plot_df["reduction_percent"].min() - 8), 0)
	ax.set_xlabel(xlabel)
	ax.set_title(f"{mitigation.capitalize()} distortion reduction")
	ax.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.7)
	ax.grid(axis="y", visible=False)

	legend_handles = [
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="",
			color=ARROW_COLORS["green"],
			markersize=6,
			label="Favorable change",
		),
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="",
			color=ARROW_COLORS["red"],
			markersize=6,
			label="Unfavorable change",
		),
	]
	ax.legend(handles=legend_handles, frameon=False, loc="lower left")

	sns.despine(ax=ax, left=False, bottom=False)
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches="tight")

	return fig, ax


def create_change_vs_stance_correlation_plot_from_significance_details(
	significance_details_df,
	correlation_df,
	mitigation,
	correlation_target=CORRELATION_TARGET_ATTRIBUTE,
	significant_only=False,
	xlabel="Average marginal effect",
	ylabel="Correlation with writer stance polarity",
	save_path=None,
):
	plot_df = build_side_effect_correlation_rows(
		significance_details_df,
		correlation_df,
		mitigation=mitigation,
		correlation_target=correlation_target,
		significant_only=significant_only,
	)

	if plot_df.empty:
		return None, None

	fig, ax = plt.subplots(figsize=(8, 6))
	point_colors = plot_df.apply(
		lambda row: "#7a7a7a"
		if not row["significant"]
		else {
			"liked": ARROW_COLORS["green"],
			"disliked": ARROW_COLORS["red"],
		}.get(row["change_group"], "#7a7a7a"),
		axis=1,
	)
	point_sizes = plot_df.apply(
		lambda row: get_change_vs_stance_marker_size(
			row["significant"],
			row["ame"],
			row["writer_ame"],
		),
		axis=1,
	)
	correlation_stats = get_plot_correlations(plot_df, x_column="ame")
	linear_fit = get_linear_fit_points(plot_df["ame"], plot_df["stance_correlation"])

	if linear_fit is not None:
		ax.plot(
			linear_fit["x_fit"],
			linear_fit["y_fit"],
			color=BEST_FIT_LINE_COLOR,
			linewidth=1.5,
			linestyle="-",
			alpha=0.9,
			zorder=2,
		)

	for row, point_color, point_size in zip(
		plot_df.itertuples(index=False),
		point_colors,
		point_sizes,
	):
		ax.plot(
			row.ame,
			row.stance_correlation,
			linestyle="",
			marker="o",
			markersize=point_size,
			color=point_color,
			markeredgewidth=0,
			alpha=0.9,
			zorder=3,
		)

	texts = []
	for _, row in plot_df.iterrows():
		texts.append(
			ax.text(
				row["ame"],
				row["stance_correlation"],
				row["attribute_label"],
				fontsize=8,
				ha="left",
				va="bottom",
			)
		)

	adjust_text(
		texts,
		ax=ax,
		force_static=(0.3, 0.5),
		force_explode=(0.3, 0.8),
		explode_radius=200,
		max_move=(20, 20),
		expand=(1.2, 1.4),
		only_move={"static": "y", "text": "xy", "explode": "xy", "pull": "xy"},
		arrowprops={"arrowstyle": "-", "color": "#7a7a7a", "lw": 0.6},
	)

	ax.axhline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
	ax.axvline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(mitigation.capitalize())

	if correlation_stats is not None:
		correlation_text = (
			f"Pearson r = {correlation_stats['pearson_r']:.2f} "
			f"(p {format_p_value(correlation_stats['pearson_p'])})\n"
			f"Spearman rho = {correlation_stats['spearman_r']:.2f} "
			f"(p {format_p_value(correlation_stats['spearman_p'])})"
		)
		ax.text(
			0.02,
			0.02,
			correlation_text,
			transform=ax.transAxes,
			ha="left",
			va="bottom",
			fontsize=9,
			bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
		)

	legend_handles = [
		Line2D(
			[0],
			[0],
			linestyle="-",
			color=BEST_FIT_LINE_COLOR,
			linewidth=1.5,
			label="Linear fit",
		),
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="",
			color=ARROW_COLORS["green"],
			markersize=6,
			label="Favorable change",
		),
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="",
			color=ARROW_COLORS["red"],
			markersize=6,
			label="Unfavorable change",
		),
		Line2D(
			[0],
			[0],
			marker="o",
			linestyle="",
			color="#7a7a7a",
			markersize=6,
			label="Insignificant change",
		),
	]
	ax.legend(handles=legend_handles, frameon=False)

	sns.despine(ax=ax)
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches="tight")

	return fig, ax


# =============================================================================
# OUTPUTS
# =============================================================================

def main():
	global DISTORTION_TOLERANCE_LOOKUP

	significance_details_df = load_significance_details()
	correlation_df = load_correlation_data()
	DISTORTION_TOLERANCE_LOOKUP = load_distortion_tolerance_lookup()

	if not significance_details_df.empty and not correlation_df.empty:
		preferred_figure_dir = os.path.join(FIGURES_DIR, PREFERRED_PARA_TYPE)
		os.makedirs(preferred_figure_dir, exist_ok=True)

		fig, _ = create_mitigation_reduction_plot(
			significance_details_df,
			mitigation="reranking",
			save_path=os.path.join(
				preferred_figure_dir,
				"reranking_distortion_reduction_vs_writer.pdf",
			),
		)
		if fig is not None:
			plt.close(fig)

		for mitigation in CHANGE_VS_STANCE_MITIGATIONS:
			fig, _ = create_change_vs_stance_correlation_plot_from_significance_details(
				significance_details_df,
				correlation_df,
				mitigation=mitigation,
				save_path=os.path.join(
					preferred_figure_dir,
					f"ame_change_vs_stance_correlation_{mitigation}_vs_writer.pdf",
				),
			)
			if fig is not None:
				plt.close(fig)


if __name__ == "__main__":
	main()
