#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 1 ANALYSIS: WRITER ENGAGEMENT
#
# Describes engagement metrics (issue knowledge, issue importance, confidence in opinion) and stance self-reported by writers in Phase 1
#
# - Merges proposition responses with proposition leaning metadata.
# - Builds histogram visualizations for engagement variables and writer stance.
# - Saves figures to `figures/main_phase_1/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from demo_paths import get_figures_dir, parse_demo_mode

DEMO_MODE = parse_demo_mode()
DATA_PATH = BASE_DIR / "data" / "main_phase_1" / "proposition_responses.csv"
PROPOSITIONS_PATH = BASE_DIR / "data" / "main_phase_1" / "propositions.csv"
FIGURES_DIR = get_figures_dir(BASE_DIR, "main_phase_1", demo_mode=DEMO_MODE)
ENGAGEMENT_OUTPUT_PATH = FIGURES_DIR / "writer_engagement_histogram.pdf"
STANCE_OUTPUT_PATH = FIGURES_DIR / "writer_stance_histogram.pdf"
HISTOGRAM_BINS = np.arange(0, 105, 5)

# Plot labels
ENGAGEMENT_LABELS = {
	"writer_knowledge": "Issue knowledge",
	"writer_importance": "Issue importance",
	"writer_confidence": "Confidence in opinion",
}

LEANING_LABELS = {
	"left": "Left-leaning propositions (n=46)",
	"right": "Right-leaning propositions (n=46)",
	"neither": "Neither-leaning propositions (n=8)",
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_phase_1_data() -> pd.DataFrame:
	responses_df = pd.read_csv(DATA_PATH)
	propositions_df = pd.read_csv(PROPOSITIONS_PATH)[["proposition_id", "proposition_leaning"]]

	return responses_df.merge(propositions_df, on="proposition_id", how="left")


# =============================================================================
# ANALYSIS
# =============================================================================

def build_long_df(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
	value_df = df[list(labels)].apply(pd.to_numeric, errors="coerce")

	return value_df.melt(
		var_name="variable",
		value_name="value",
	).dropna(subset=["value"])


def draw_summary_lines(ax: plt.Axes, values: pd.Series) -> None:
	mean_value = values.mean()
	median_value = values.median()

	ax.axvline(
		mean_value,
		color="black",
		linestyle="--",
		linewidth=1.2,
		alpha=0.8,
	)
	ax.annotate(
		f"Mean {mean_value:.1f}",
		xy=(mean_value, 0.94),
		xycoords=("data", "axes fraction"),
		xytext=(-3, 0),
		textcoords="offset points",
		ha="right",
		va="top",
		fontsize=10,
	)

	ax.axvline(
		median_value,
		color="black",
		linestyle=":",
		linewidth=1.2,
		alpha=0.8,
	)
	ax.annotate(
		f"Median {median_value:.1f}",
		xy=(median_value, 0.94),
		xycoords=("data", "axes fraction"),
		xytext=(3, 0),
		textcoords="offset points",
		ha="left",
		va="top",
		fontsize=10,
	)


# =============================================================================
# OUTPUTS
# =============================================================================

def save_histogram_row_plot(
	plot_series: list[tuple[str, pd.Series]],
	output_path: Path,
	*,
	xlabel: str,
) -> None:
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, len(plot_series), figsize=(12, 4), sharey=True)
	for ax, (label, values) in zip(axes, plot_series, strict=True):
		clean_values = pd.to_numeric(values, errors="coerce").dropna()

		ax.hist(
			clean_values,
			bins=HISTOGRAM_BINS,
			density=True,
			color="#838383",
			edgecolor="#838383",
			linewidth=0.8,
		)

		draw_summary_lines(ax, clean_values)
		ax.set_xlim(0, 100)
		ax.set_xticks(np.arange(0, 101, 20))
		ax.set_xlabel(xlabel)
		ax.set_title(label)
		ax.spines["right"].set_visible(False)
		ax.spines["top"].set_visible(False)

	axes[0].set_ylabel("Density")
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def save_writer_engagement_histogram(df: pd.DataFrame) -> None:
	plot_df = build_long_df(df, ENGAGEMENT_LABELS)
	plot_series = [
		(label, plot_df.loc[plot_df["variable"] == variable, "value"])
		for variable, label in ENGAGEMENT_LABELS.items()
	]
	save_histogram_row_plot(
		plot_series,
		ENGAGEMENT_OUTPUT_PATH,
		xlabel="Writer response (0-100)",
	)


def save_writer_stance_pre_by_leaning_histogram(df: pd.DataFrame) -> None:
	plot_series = [
		(label, df.loc[df["proposition_leaning"] == leaning, "writer_stance_pre"])
		for leaning, label in LEANING_LABELS.items()
	]
	save_histogram_row_plot(
		plot_series,
		STANCE_OUTPUT_PATH,
		xlabel="Writer's stated issue stance (0-100)",
	)


def main() -> None:
	phase_1_df = load_phase_1_data()
	save_writer_engagement_histogram(phase_1_df)
	save_writer_stance_pre_by_leaning_histogram(phase_1_df)
	print(f"Saved writer engagement histogram to {ENGAGEMENT_OUTPUT_PATH}")
	print(f"Saved writer stance histogram to {STANCE_OUTPUT_PATH}")


if __name__ == "__main__":
	main()
