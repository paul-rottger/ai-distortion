#!/usr/bin/env python3

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import sys
from pathlib import Path

import pandas as pd

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_BASE_DIR = BASE_DIR / "results" / "main_phase_2_distortion"
SUMMARY_DIR_NAME = "summary"

# Internal imports
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))
from variable_definitions import SCALE_ATTRIBUTES, ORDINAL_VARS as ORDINAL_ATTRIBUTES

# Analysis configuration
DATA_SPLITS = ["preferred", "edited", "unedited"]

INPUT_CONDITION_TERMS = [
	"input_condition_bullets-based",
	"input_condition_improve",
	"input_condition_rewrite",
	"input_condition_stance-based",
]

VARIABLE_GROUPS = {
	"scale": {
		"attributes": SCALE_ATTRIBUTES,
		"metric_column": "ame",
		"summary_metric": "average_absolute_ame",
		"null_value": 0.0,
		"zero_label": "zero AME(s)",
		"direction_label": "AME directions",
	},
	"ordinal": {
		"attributes": ORDINAL_ATTRIBUTES,
		"metric_column": "odds_ratio",
		"summary_metric": "average_odds_ratio",
		"null_value": 1.0,
		"zero_label": "unit odds ratio(s)",
		"direction_label": "odds-ratio directions",
	},
}


# =============================================================================
# LOAD DATA
# =============================================================================

def resolve_results_path(data_split: str, attribute: str) -> Path:
	candidates = [
		RESULTS_BASE_DIR / data_split / f"{attribute}_by_input_condition.csv",
		RESULTS_BASE_DIR / data_split / f"{attribute}_by_input.csv",
	]

	for file_path in candidates:
		if file_path.exists():
			return file_path

	raise FileNotFoundError(
		f"Missing regression results for {attribute} in split '{data_split}'. Tried: "
		+ ", ".join(str(path) for path in candidates)
	)


def load_attribute_results(data_split: str, attribute: str, metric_column: str) -> pd.DataFrame:
	file_path = resolve_results_path(data_split, attribute)
	df = pd.read_csv(file_path)

	required_columns = {"term", metric_column}
	if "p" in df.columns:
		p_column = "p"
	elif "p_value" in df.columns:
		p_column = "p_value"
	else:
		raise ValueError(f"No p-value column found in {file_path}")

	missing_columns = required_columns - set(df.columns)
	if missing_columns:
		raise ValueError(f"Missing columns {sorted(missing_columns)} in {file_path}")

	input_df = df[df["term"].isin(INPUT_CONDITION_TERMS)].copy()
	if len(input_df) != len(INPUT_CONDITION_TERMS):
		raise ValueError(
			f"Expected {len(INPUT_CONDITION_TERMS)} input-condition rows in {file_path}, found {len(input_df)}"
		)

	input_df["p_selected"] = input_df[p_column]
	return input_df[["term", "p_selected", metric_column]].sort_values("term")


# =============================================================================
# ANALYSIS
# =============================================================================

def summarize_direction(metric_series: pd.Series, null_value: float) -> pd.Series:
	if null_value == 0.0:
		return metric_series.map(
			lambda value: "positive" if value > 0 else "negative" if value < 0 else "null"
		)

	return metric_series.map(
		lambda value: "positive" if value > null_value else "negative" if value < null_value else "null"
	)


def evaluate_attribute(
	input_df: pd.DataFrame,
	metric_column: str,
	null_value: float,
	zero_label: str,
	direction_label: str,
	alpha: float = 0.05,
) -> tuple[bool, str | None]:
	failed_reasons: list[str] = []

	nonsignificant_terms = input_df.loc[input_df["p_selected"] >= alpha, "term"].tolist()
	if nonsignificant_terms:
		failed_reasons.append(
			"non-significant coefficient(s): " + ", ".join(nonsignificant_terms)
		)

	null_terms = input_df.loc[input_df[metric_column] == null_value, "term"].tolist()
	if null_terms:
		failed_reasons.append(f"{zero_label}: " + ", ".join(null_terms))

	directions = summarize_direction(input_df[metric_column], null_value)
	non_null_directions = set(directions[directions != "null"])
	if len(non_null_directions) > 1:
		direction_summary = ", ".join(
			f"{row.term}={direction}"
			for row, direction in zip(input_df.itertuples(), directions, strict=False)
		)
		failed_reasons.append(f"mixed {direction_label}: " + direction_summary)

	if failed_reasons:
		return False, "; ".join(failed_reasons)

	return True, None


def build_group_summary(data_split: str, variable_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	group_config = VARIABLE_GROUPS[variable_type]
	metric_column = group_config["metric_column"]
	attribute_results: list[pd.DataFrame] = []
	attribute_statuses: list[dict[str, str | bool]] = []

	for attribute in group_config["attributes"]:
		try:
			input_df = load_attribute_results(data_split, attribute, metric_column)
		except FileNotFoundError as exc:
			attribute_statuses.append(
				{
					"split": data_split,
					"analysis_group": variable_type,
					"attribute": attribute,
					"in_summary": False,
					"passes_check": False,
					"status_note": str(exc),
				}
			)
			continue

		input_df["attribute"] = attribute
		input_df["split"] = data_split
		input_df["variable_type"] = variable_type
		attribute_results.append(input_df)

		matches, failure_reason = evaluate_attribute(
			input_df,
			metric_column=metric_column,
			null_value=group_config["null_value"],
			zero_label=group_config["zero_label"],
			direction_label=group_config["direction_label"],
		)
		attribute_statuses.append(
			{
				"split": data_split,
				"analysis_group": variable_type,
				"attribute": attribute,
				"in_summary": True,
				"passes_check": matches,
				"status_note": failure_reason or "",
			}
		)

	if attribute_results:
		combined_results = pd.concat(attribute_results, ignore_index=True)
		summary_values = combined_results[metric_column].abs() if variable_type == "scale" else combined_results[metric_column]
		summary_table = (
			combined_results.assign(
				input_condition=lambda df: df["term"].str.removeprefix("input_condition_"),
				summary_value=summary_values,
			)
			.groupby(["split", "variable_type", "input_condition"], as_index=False)
			.agg(
				n_attributes=("attribute", "nunique"),
				average_effect=("summary_value", "mean"),
			)
			.assign(summary_metric=group_config["summary_metric"])
			.sort_values("average_effect", ascending=False)
		)
	else:
		summary_table = pd.DataFrame(
			columns=["split", "variable_type", "input_condition", "n_attributes", "average_effect", "summary_metric"]
		)

	return summary_table, pd.DataFrame(attribute_statuses)


# =============================================================================
# OUTPUTS
# =============================================================================

def main() -> None:
	for data_split in DATA_SPLITS:
		summary_tables: list[pd.DataFrame] = []
		attribute_status_tables: list[pd.DataFrame] = []

		for variable_type in VARIABLE_GROUPS:
			summary_table, status_table = build_group_summary(data_split, variable_type)
			summary_tables.append(summary_table)
			attribute_status_tables.append(status_table)

		combined_summary = pd.concat(summary_tables, ignore_index=True)
		combined_status = pd.concat(attribute_status_tables, ignore_index=True)

		if not combined_summary.empty:
			combined_summary["average_effect"] = combined_summary["average_effect"].round(3)

		output_dir = RESULTS_BASE_DIR / data_split / SUMMARY_DIR_NAME
		output_dir.mkdir(parents=True, exist_ok=True)

		summary_output_path = output_dir / "phase_2_distortion_by_input_condition_summary.csv"
		status_output_path = output_dir / "phase_2_distortion_by_input_condition_attribute_status.csv"

		combined_summary.to_csv(summary_output_path, index=False)
		combined_status.to_csv(status_output_path, index=False)

		print(f"Split: {data_split}")
		for variable_type, group_config in VARIABLE_GROUPS.items():
			status_subset = combined_status[combined_status["analysis_group"] == variable_type]
			included_status = status_subset[status_subset["in_summary"]]
			matching_attributes = included_status.loc[included_status["passes_check"], "attribute"].tolist()
			failing_attributes = included_status.loc[~included_status["passes_check"], ["attribute", "status_note"]]
			missing_attributes = status_subset.loc[~status_subset["in_summary"], "attribute"].tolist()

			print(
				f"Attributes where all four input-condition coefficients are significant and {group_config['direction_label']} align "
				f"for {variable_type} variables: {len(matching_attributes)} / {len(included_status)}"
			)

			if matching_attributes:
				print("Matching attributes:")
				for attribute in matching_attributes:
					print(attribute)

			if not failing_attributes.empty:
				print("Failing attributes:")
				for row in failing_attributes.itertuples(index=False):
					print(f"{row.attribute}: {row.status_note}")

			if missing_attributes:
				print("Missing attributes:")
				for attribute in missing_attributes:
					print(attribute)

		print("Combined input-condition summary:")
		print(combined_summary.to_string(index=False))
		print(f"Wrote {summary_output_path}")
		print(f"Wrote {status_output_path}")


if __name__ == "__main__":
	main()