from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "main_phase_1" / "proposition_responses.csv"
FIGURES_DIR = BASE_DIR / "figures" / "main_phase_1"
RESULTS_DIR = BASE_DIR / "results" / "main_phase_1"


def parse_made_edits(value: object) -> bool:
	if pd.isna(value):
		return False

	normalized = str(value).strip().lower()
	return normalized in {"1", "true", "t", "yes"}


def build_writer_edit_count_table(df: pd.DataFrame) -> pd.DataFrame:
	writer_edit_counts = (
		df.assign(made_edits_flag=df["made_edits"].map(parse_made_edits).astype(int))
		.groupby("writer_id", as_index=False)["made_edits_flag"]
		.sum()
		.rename(columns={"made_edits_flag": "edited_paragraph_count"})
	)

	counts = (
		writer_edit_counts["edited_paragraph_count"]
		.value_counts()
		.reindex(range(4), fill_value=0)
		.rename_axis("edited_paragraph_count")
		.reset_index(name="n_writers")
	)
	counts["edited_paragraphs_label"] = counts["edited_paragraph_count"].map(lambda value: f"{value} of 3")

	return counts[["edited_paragraph_count", "edited_paragraphs_label", "n_writers"]]


def save_histogram(
	values: pd.Series,
	output_path: Path,
	*,
	xlabel: str,
	bins: int,
	xlim: tuple[float, float] | None = None,
) -> None:
	mean_value = values.mean()
	median_value = values.median()

	fig, ax = plt.subplots(figsize=(4.5, 3))
	ax.hist(values, bins=bins, edgecolor=None, color="#808080", alpha=1)
	ax.axvline(mean_value, color="#000000", linestyle="-", alpha=1, label=f"Mean: {mean_value:.2f}")
	ax.axvline(
		median_value,
		color="#000000",
		linestyle="--",
		alpha=1,
		label=f"Median: {median_value:.2f}",
	)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Responses")
	ax.spines["right"].set_visible(False)
	ax.spines["top"].set_visible(False)

	if xlim is not None:
		ax.set_xlim(*xlim)

	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def save_histograms(edited_df: pd.DataFrame) -> None:
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)

	distance_max = int(edited_df["levenshtein_distance"].max())
	distance_bins = min(50, max(10, distance_max))

	save_histogram(
		edited_df["levenshtein_distance"],
		FIGURES_DIR / "paragraph_edits_levenshtein_distance_histogram.pdf",
		xlabel="levenshtein_distance (higher means more edits)",
		bins=distance_bins,
	)
	save_histogram(
		edited_df["levenshtein_ratio"],
		FIGURES_DIR / "paragraph_edits_levenshtein_ratio_histogram.pdf",
		xlabel="levenshtein_ratio (0-1 scale, where 1 is identical)",
		bins=50,
		xlim=(0, 1),
	)


def load_responses() -> pd.DataFrame:
	return pd.read_csv(DATA_PATH)


def build_edit_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
	edit_metrics_df = df.copy()
	edit_metrics_df["made_edits_flag"] = edit_metrics_df["made_edits"].map(parse_made_edits)
	edit_metrics_df["model_paragraph"] = edit_metrics_df["model_paragraph"].fillna("")
	edit_metrics_df["edited_paragraph"] = edit_metrics_df["edited_paragraph"].fillna("")
	edit_metrics_df["edit_ratio"] = 0.0
	edit_metrics_df["edit_distance"] = 0

	edited_mask = edit_metrics_df["made_edits_flag"]
	model_paragraphs = edit_metrics_df.loc[edited_mask, "model_paragraph"]
	edited_paragraphs = edit_metrics_df.loc[edited_mask, "edited_paragraph"]

	edit_metrics_df.loc[edited_mask, "edit_ratio"] = [
		levenshtein_ratio(model_paragraph, edited_paragraph)
		for model_paragraph, edited_paragraph in zip(
			model_paragraphs,
			edited_paragraphs,
			strict=False,
		)
	]
	edit_metrics_df.loc[edited_mask, "edit_distance"] = [
		levenshtein_distance(model_paragraph, edited_paragraph)
		for model_paragraph, edited_paragraph in zip(
			model_paragraphs,
			edited_paragraphs,
			strict=False,
		)
	]

	return edit_metrics_df[
		[
			"writer_id",
			"proposition_id",
			"proposition_number",
			"made_edits_flag",
			"edit_ratio",
			"edit_distance",
		]
	].rename(columns={"made_edits_flag": "made_edits"})


def load_edited_responses(df: pd.DataFrame) -> pd.DataFrame:
	edited_df = df.loc[df["made_edits"].map(parse_made_edits)].copy()
	edited_df["model_paragraph"] = edited_df["model_paragraph"].fillna("")
	edited_df["edited_paragraph"] = edited_df["edited_paragraph"].fillna("")
	edited_df = edited_df.loc[
		(edited_df["model_paragraph"] != "") | (edited_df["edited_paragraph"] != "")
	].copy()

	distances: list[int] = []
	ratios: list[float] = []

	for row in edited_df.itertuples(index=False):
		distance = levenshtein_distance(row.model_paragraph, row.edited_paragraph)
		distances.append(distance)
		ratios.append(levenshtein_ratio(row.model_paragraph, row.edited_paragraph))

	edited_df["model_paragraph_length"] = edited_df["model_paragraph"].str.len()
	edited_df["edited_paragraph_length"] = edited_df["edited_paragraph"].str.len()
	edited_df["levenshtein_distance"] = distances
	edited_df["levenshtein_ratio"] = ratios

	return edited_df


def write_outputs(
	df: pd.DataFrame,
	edited_df: pd.DataFrame,
	edit_metrics_table: pd.DataFrame,
) -> pd.DataFrame:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	save_histograms(edited_df)
	edit_metrics_table.to_csv(
		RESULTS_DIR / "paragraph_edit_ratio_by_writer_proposition.csv",
		index=False,
	)

	writer_edit_count_table = build_writer_edit_count_table(df)
	writer_edit_count_table.to_csv(
		RESULTS_DIR / "paragraph_edit_counts_by_writer.csv",
		index=False,
	)

	return writer_edit_count_table


def print_console_summary(edited_df: pd.DataFrame, writer_edit_count_table: pd.DataFrame) -> None:
	distance_series = edited_df["levenshtein_distance"]
	ratio_series = edited_df["levenshtein_ratio"]

	print(f"Edited responses analyzed: {len(edited_df)}")
	print(
		"Levenshtein distance "
		f"mean={distance_series.mean():.2f}, "
		f"median={distance_series.median():.2f}, "
		f"min={distance_series.min():.0f}, "
		f"max={distance_series.max():.0f}"
	)
	print(
		"Levenshtein ratio "
		f"mean={ratio_series.mean():.4f}, "
		f"median={ratio_series.median():.4f}, "
		f"min={ratio_series.min():.4f}, "
		f"max={ratio_series.max():.4f}"
	)
	print("Writer edit counts:")
	for row in writer_edit_count_table.itertuples(index=False):
		print(f"  {row.edited_paragraphs_label}: {row.n_writers}")


def main() -> None:
	df = load_responses()
	edit_metrics_table = build_edit_metrics_table(df)
	edited_df = load_edited_responses(df)
	writer_edit_count_table = write_outputs(df, edited_df, edit_metrics_table)
	print_console_summary(edited_df, writer_edit_count_table)


if __name__ == "__main__":
	main()
