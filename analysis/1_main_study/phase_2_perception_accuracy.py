#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 2 ANALYSIS: PERCEPTION ACCURACY
#
# Compares reader perceptions of writers (Phase 2) with writers' own
# self-reports (Phase 1), for writer and AI paragraphs.
#
# - Joins per-proposition self-reports (stance, knowledge, importance,
#   confidence, affect) and writer demographics to each reader rating.
# - Builds the standard unedited, edited, and preferred data splits.
# - Scale attributes: correlation (tracking), mean signed error (bias), and
#   mean absolute error between perceived and self-reported values.
# - Categorical attributes: accuracy, chance accuracy, and Cohen's kappa, plus
#   the correlation of ordinal codes for ordered attributes.
# - Computes metrics at rater level and at paragraph level (aggregated across
#   raters), with writer-clustered bootstrap CIs for each paragraph type and
#   for the model - writer difference.
# - Breaks down model - writer differences by model input condition, and
#   checks robustness of stance results to the self-report reference point.
# - Exports tables to `results/main_phase_2_accuracy/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis"))
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from data_aggregation import modal_value
from demo_paths import get_results_dir, parse_demo_mode
from variable_definitions import CATEGORICAL_LEVELS

DEMO_MODE = parse_demo_mode()
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = get_results_dir(BASE_DIR, "main_phase_2_accuracy", demo_mode=DEMO_MODE)

SEED = 123
N_BOOTSTRAP = 100 if DEMO_MODE else 1000
PARAGRAPH_TYPES = ["writer", "model"]
DIFFERENCE_LABEL = "model - writer"
PARAGRAPH_KEYS = ["writer_id", "proposition_id", "paragraph_type"]
INPUT_CONDITIONS = ["stance-based", "bullets-based", "rewrite", "improve"]
STANCE_REFERENCES = ["pre", "post", "final"]
PRIMARY_STANCE_REFERENCE = "post"  # reported right after writing, before seeing the AI paragraph
AGE_BINS = [0, 29, 39, 49, 59, 69, np.inf]

# Perceived attribute -> self-reported column
SCALE_PAIRS = {
	"writer_stance": "self_stance",
	"writer_stance_polarity": "self_stance_polarity",
	"writer_knowledge": "self_knowledge",
	"writer_importance": "self_importance",
	"writer_confidence": "self_confidence",
	"writer_affect_x": "self_affect_x",
	"writer_affect_y": "self_affect_y",
}


def categorical_spec(self_column: str, levels: list[str], ordered: bool = True) -> dict:
	return {
		"self_column": self_column,
		"levels": levels,
		"ordered_levels": [level for level in levels if level != "Other"] if ordered else None,
	}


# Perceived attribute -> self-reported column, all levels, and ordered levels ("Other" excluded)
CATEGORICAL_PAIRS = {
	"writer_age_binned": categorical_spec("self_age_binned", CATEGORICAL_LEVELS["writer_age_binned"]),
	"writer_education": categorical_spec("self_education", CATEGORICAL_LEVELS["writer_education"] + ["Other"]),
	"writer_income": categorical_spec("self_income", CATEGORICAL_LEVELS["writer_income"]),
	"writer_english_skills": categorical_spec("self_englishSkills", CATEGORICAL_LEVELS["writer_english_skills"]),
	"writer_politicalIdeology": categorical_spec("self_politicalIdeology", CATEGORICAL_LEVELS["writer_politicalIdeology"]),
	"writer_english_first": categorical_spec("self_englishFirst", CATEGORICAL_LEVELS["writer_english_first"], ordered=False),
	"writer_gender": categorical_spec("self_gender", CATEGORICAL_LEVELS["writer_gender"], ordered=False),
	"writer_race": categorical_spec("self_race", CATEGORICAL_LEVELS["writer_race"], ordered=False),
	"writer_politicalParty": categorical_spec("self_politicalParty", CATEGORICAL_LEVELS["writer_politicalParty"], ordered=False),
}

SELF_REPORT_COLUMNS = {
	"writer_knowledge": "self_knowledge",
	"writer_importance": "self_importance",
	"writer_confidence": "self_confidence",
	"writer_affect_x": "self_affect_x",
	"writer_affect_y": "self_affect_y",
	"model_input_condition": "input_condition",
	**{f"writer_stance_{reference}": f"self_stance_{reference}" for reference in STANCE_REFERENCES},
}

DEMOGRAPHIC_COLUMNS = {
	"age": "self_age",
	"gender": "self_gender",
	"race": "self_race",
	"englishFirst": "self_englishFirst",
	"englishSkills": "self_englishSkills",
	"education": "self_education",
	"income": "self_income",
	"politicalParty": "self_politicalParty",
	"politicalIdeology": "self_politicalIdeology",
}


def code_column(column: str) -> str:
	return f"{column}_code"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_self_reports() -> pd.DataFrame:
	responses = pd.read_csv(DATA_DIR / "main_phase_1" / "proposition_responses.csv")
	self_reports = responses[["writer_id", "proposition_id", "writer_preference", *SELF_REPORT_COLUMNS]].rename(columns=SELF_REPORT_COLUMNS)

	# Polarity mirrors the reader-side definition: distance from the scale midpoint, rescaled to 0-100
	for reference in STANCE_REFERENCES:
		self_reports[f"self_stance_polarity_{reference}"] = (self_reports[f"self_stance_{reference}"] - 50).abs() * 2
	self_reports["self_stance"] = self_reports[f"self_stance_{PRIMARY_STANCE_REFERENCE}"]
	self_reports["self_stance_polarity"] = self_reports[f"self_stance_polarity_{PRIMARY_STANCE_REFERENCE}"]
	return self_reports


def load_demographics() -> pd.DataFrame:
	participants = pd.read_csv(DATA_DIR / "main_phase_1" / "participants.csv")
	demographics = participants[["writer_id", *DEMOGRAPHIC_COLUMNS]].rename(columns=DEMOGRAPHIC_COLUMNS)

	demographics = demographics.replace("Prefer not to say", np.nan)
	# Readers could only choose Yes or No for English as a first language
	demographics["self_englishFirst"] = demographics["self_englishFirst"].replace("Bilingual from birth", "Yes")
	demographics["self_age_binned"] = pd.cut(
		demographics["self_age"], bins=AGE_BINS, labels=CATEGORICAL_LEVELS["writer_age_binned"]
	).astype(object)
	return demographics


def add_ordinal_codes(data: pd.DataFrame) -> pd.DataFrame:
	for perceived, spec in CATEGORICAL_PAIRS.items():
		if spec["ordered_levels"] is None:
			continue
		codes = {level: index for index, level in enumerate(spec["ordered_levels"])}
		data[code_column(perceived)] = data[perceived].map(codes)
		data[code_column(spec["self_column"])] = data[spec["self_column"]].map(codes)
	return data


def load_data() -> pd.DataFrame:
	annotations = pd.read_csv(DATA_DIR / "main_phase_2" / "annotations.csv")
	data = (
		annotations
		.merge(load_self_reports(), on=["writer_id", "proposition_id"], how="left", validate="many_to_one")
		.merge(load_demographics(), on="writer_id", how="left", validate="many_to_one")
	)
	return add_ordinal_codes(data)


def build_splits(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
	"""Mirrors `load_phase2_splits` in analysis/utils_r/data_loading.R."""
	pair_keys = data["writer_id"] + "_" + data["proposition_id"].astype(str)
	edited_keys = set(pair_keys[data["paragraph_type"] == "edited"])

	unedited = data[data["paragraph_type"].isin(PARAGRAPH_TYPES)]
	edited = data[~((data["paragraph_type"] == "model") & pair_keys.isin(edited_keys))].assign(
		paragraph_type=lambda d: d["paragraph_type"].replace("edited", "model")
	)
	preferred = edited[edited["writer_preference"] != "original"]

	return {"unedited": unedited, "edited": edited, "preferred": preferred}


def aggregate_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
	"""Paragraph-level perceptions: mean for scale and ordinal codes, modal value for categories."""
	aggregation_map = {"input_condition": "first"}
	for perceived, self_column in SCALE_PAIRS.items():
		aggregation_map[perceived] = "mean"
		aggregation_map[self_column] = "first"
	for perceived, spec in CATEGORICAL_PAIRS.items():
		aggregation_map[perceived] = modal_value
		aggregation_map[spec["self_column"]] = "first"
		if spec["ordered_levels"] is not None:
			aggregation_map[code_column(perceived)] = "mean"
			aggregation_map[code_column(spec["self_column"])] = "first"

	return df.groupby(PARAGRAPH_KEYS, sort=False).agg(aggregation_map).reset_index()


# =============================================================================
# ANALYSIS
# =============================================================================

# Per-writer sufficient statistics are summed with bootstrap weights (writer
# resampling counts), so every resample is a single matrix product. Each stats
# table has a leading "n" column with the number of ratings.

def scale_stats(df: pd.DataFrame, perceived: str, self_column: str) -> pd.DataFrame:
	d = df[["writer_id", perceived, self_column]].dropna()
	x, y = d[perceived], d[self_column]
	parts = pd.DataFrame({
		"writer_id": d["writer_id"],
		"n": 1.0,
		"x": x,
		"y": y,
		"xx": x * x,
		"yy": y * y,
		"xy": x * y,
		"error": x - y,
		"abs_error": (x - y).abs(),
	})
	return parts.groupby("writer_id").sum()


def scale_metrics(stats: np.ndarray) -> dict[str, np.ndarray]:
	n, x, y, xx, yy, xy, error, abs_error = np.moveaxis(stats, -1, 0)
	with np.errstate(divide="ignore", invalid="ignore"):
		mean_x, mean_y = x / n, y / n
		covariance = xy / n - mean_x * mean_y
		variance_x = xx / n - mean_x**2
		variance_y = yy / n - mean_y**2
		return {
			"correlation": covariance / np.sqrt(variance_x * variance_y),
			"bias": error / n,
			"mae": abs_error / n,
		}


def ordinal_metrics(stats: np.ndarray) -> dict[str, np.ndarray]:
	return {"ordinal_correlation": scale_metrics(stats)["correlation"]}


def categorical_stats(df: pd.DataFrame, perceived: str, self_column: str, levels: list[str]) -> pd.DataFrame:
	d = df[["writer_id", perceived, self_column]].dropna()
	d = d[d[perceived].isin(levels) & d[self_column].isin(levels)]
	n_levels = len(levels)
	# Flattened contingency table: rows are self-reported levels, columns perceived levels
	cells = (
		pd.Categorical(d[self_column], categories=levels).codes * n_levels
		+ pd.Categorical(d[perceived], categories=levels).codes
	)
	counts = pd.crosstab(d["writer_id"].to_numpy(), cells).reindex(columns=range(n_levels**2), fill_value=0)
	counts.insert(0, "n", counts.sum(axis=1))
	return counts


def categorical_metrics(stats: np.ndarray) -> dict[str, np.ndarray]:
	n = stats[..., 0]
	n_levels = int(round(np.sqrt(stats.shape[-1] - 1)))
	table = stats[..., 1:].reshape(stats.shape[:-1] + (n_levels, n_levels))
	with np.errstate(divide="ignore", invalid="ignore"):
		accuracy = np.trace(table, axis1=-2, axis2=-1) / n
		chance_accuracy = (table.sum(axis=-1) * table.sum(axis=-2)).sum(axis=-1) / n**2
		return {
			"accuracy": accuracy,
			"chance_accuracy": chance_accuracy,
			"kappa": (accuracy - chance_accuracy) / (1 - chance_accuracy),
		}


def build_measures(scale_pairs: dict[str, str], include_categorical: bool = True) -> list[dict]:
	measures = [
		{
			"attribute": perceived,
			"kind": "scale",
			"stats": partial(scale_stats, perceived=perceived, self_column=self_column),
			"metrics": scale_metrics,
		}
		for perceived, self_column in scale_pairs.items()
	]
	if not include_categorical:
		return measures

	for perceived, spec in CATEGORICAL_PAIRS.items():
		measures.append({
			"attribute": perceived,
			"kind": "categorical",
			"stats": partial(categorical_stats, perceived=perceived, self_column=spec["self_column"], levels=spec["levels"]),
			"metrics": categorical_metrics,
		})
		if spec["ordered_levels"] is not None:
			measures.append({
				"attribute": perceived,
				"kind": "categorical",
				"stats": partial(scale_stats, perceived=code_column(perceived), self_column=code_column(spec["self_column"])),
				"metrics": ordinal_metrics,
			})
	return measures


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
	low, high = np.nanpercentile(values, [2.5, 97.5])
	return float(low), float(high)


def bootstrap_p(values: np.ndarray) -> float:
	values = values[~np.isnan(values)]
	return float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))


def evaluate(df: pd.DataFrame, measure: dict, rng: np.random.Generator) -> list[dict]:
	"""Point estimates and writer-clustered bootstrap CIs per paragraph type and for the difference."""
	writer_stats = {
		paragraph_type: measure["stats"](df[df["paragraph_type"] == paragraph_type])
		for paragraph_type in PARAGRAPH_TYPES
	}
	writer_index = writer_stats["writer"].index.union(writer_stats["model"].index)
	n_writers = len(writer_index)
	weights = rng.multinomial(n_writers, np.full(n_writers, 1 / n_writers), size=N_BOOTSTRAP).astype(float)

	rows, points, boots = [], {}, {}
	for paragraph_type, stats_df in writer_stats.items():
		stats = stats_df.reindex(writer_index, fill_value=0).to_numpy(dtype=float)
		points[paragraph_type] = measure["metrics"](stats.sum(axis=0))
		boots[paragraph_type] = measure["metrics"](weights @ stats)
		for metric, estimate in points[paragraph_type].items():
			ci_low, ci_high = percentile_interval(boots[paragraph_type][metric])
			rows.append({
				"paragraph_type": paragraph_type,
				"metric": metric,
				"estimate": float(estimate),
				"ci_low": ci_low,
				"ci_high": ci_high,
				"n_ratings": int(stats_df["n"].sum()),
				"n_writers": len(stats_df),
			})

	for metric in points["writer"]:
		boot_difference = boots["model"][metric] - boots["writer"][metric]
		ci_low, ci_high = percentile_interval(boot_difference)
		rows.append({
			"paragraph_type": DIFFERENCE_LABEL,
			"metric": metric,
			"estimate": float(points["model"][metric] - points["writer"][metric]),
			"ci_low": ci_low,
			"ci_high": ci_high,
			"p_bootstrap": bootstrap_p(boot_difference),
			"n_writers": n_writers,
		})

	return [{"attribute": measure["attribute"], "kind": measure["kind"], **row} for row in rows]


def evaluate_all(df: pd.DataFrame, measures: list[dict], rng: np.random.Generator, **labels) -> list[dict]:
	return [{**labels, **row} for measure in measures for row in evaluate(df, measure, rng)]


def confusion_matrices(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
	tables = []
	for split, df in splits.items():
		for paragraph_type in PARAGRAPH_TYPES:
			type_df = df[df["paragraph_type"] == paragraph_type]
			for perceived, spec in CATEGORICAL_PAIRS.items():
				levels = spec["levels"]
				table = (
					type_df[[spec["self_column"], perceived]]
					.set_axis(["self_reported", "perceived"], axis=1)
					.value_counts()
					.reindex(pd.MultiIndex.from_product([levels, levels], names=["self_reported", "perceived"]), fill_value=0)
					.rename("n")
					.reset_index()
				)
				table["row_share"] = table["n"] / table.groupby("self_reported", observed=False)["n"].transform("sum")
				tables.append(table.assign(split=split, paragraph_type=paragraph_type, attribute=perceived))

	columns = ["split", "paragraph_type", "attribute", "self_reported", "perceived", "n", "row_share"]
	return pd.concat(tables, ignore_index=True)[columns]


# =============================================================================
# OUTPUTS
# =============================================================================

def save_table(df: pd.DataFrame, file_name: str) -> None:
	path = RESULTS_DIR / file_name
	if "n_ratings" in df:
		df = df.astype({"n_ratings": "Int64"})
	df.to_csv(path, index=False)
	print(f"Wrote {path.relative_to(BASE_DIR)}")


def print_overview(results: pd.DataFrame) -> None:
	overview = results[(results["split"] == "unedited") & (results["level"] == "rater")]
	print("\nUnedited split, rater level (estimate by paragraph type):")
	print(
		overview.pivot_table(index=["kind", "attribute", "metric"], columns="paragraph_type", values="estimate", sort=False)
		.round(3)
		.to_string()
	)


def main() -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	rng = np.random.default_rng(SEED)

	splits = build_splits(load_data())
	measures = build_measures(SCALE_PAIRS)

	# Main accuracy metrics by split and level
	rows = []
	for split, df in splits.items():
		for level, level_df in (("rater", df), ("paragraph", aggregate_paragraphs(df))):
			print(f"Evaluating {split} split at {level} level ({len(level_df)} rows)")
			rows += evaluate_all(level_df, measures, rng, split=split, level=level)
	results = pd.DataFrame(rows)
	is_difference = results["paragraph_type"] == DIFFERENCE_LABEL
	for kind in ("scale", "categorical"):
		save_table(results[(results["kind"] == kind) & ~is_difference].drop(columns="p_bootstrap"), f"{kind}_accuracy.csv")
		save_table(results[(results["kind"] == kind) & is_difference].drop(columns="n_ratings"), f"{kind}_accuracy_differences.csv")

	# Model - writer differences by input condition (rater level, unedited split)
	rows = []
	for input_condition in INPUT_CONDITIONS:
		print(f"Evaluating input condition: {input_condition}")
		condition_df = splits["unedited"][splits["unedited"]["input_condition"] == input_condition]
		rows += evaluate_all(condition_df, measures, rng, split="unedited", level="rater", input_condition=input_condition)
	save_table(pd.DataFrame(rows), "by_input_condition.csv")

	# Stance results with pre-writing, post-writing, and final self-reports as reference
	rows = []
	for reference in STANCE_REFERENCES:
		reference_measures = build_measures(
			{
				"writer_stance": f"self_stance_{reference}",
				"writer_stance_polarity": f"self_stance_polarity_{reference}",
			},
			include_categorical=False,
		)
		for split, df in splits.items():
			rows += evaluate_all(df, reference_measures, rng, split=split, level="rater", stance_reference=reference)
	save_table(pd.DataFrame(rows), "stance_reference_robustness.csv")

	save_table(confusion_matrices(splits), "confusion_matrices.csv")
	print_overview(results)


if __name__ == "__main__":
	main()
