from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "main_phase_2_distortion" / "preferred"

SCALE_ATTRIBUTES = [
	"paragraph_formality",
	"paragraph_clarity",
	"paragraph_informativeness",
	"paragraph_originality",
	"paragraph_relevance",
	"writer_knowledge",
	"writer_importance",
	"writer_confidence",
	"writer_stance_polarity",
	"paragraph_hope",
	"paragraph_excitement",
	"paragraph_fear",
	"paragraph_disgust",
	"paragraph_anger",
	"writer_affect_x",
	"writer_affect_y",
	"writer_optimism",
	"writer_community",
	"writer_friendliness",
	"writer_openness",
]

INPUT_CONDITION_TERMS = [
	"input_condition_bullets-based",
	"input_condition_improve",
	"input_condition_rewrite",
	"input_condition_stance-based",
]


def resolve_results_path(attribute: str) -> Path:
	candidates = [
		RESULTS_DIR / f"{attribute}_by_input_condition.csv",
		RESULTS_DIR / f"{attribute}_by_input.csv",
	]

	for file_path in candidates:
		if file_path.exists():
			return file_path

	raise FileNotFoundError(
		f"Missing regression results for {attribute}. Tried: "
		+ ", ".join(str(path) for path in candidates)
	)


def load_attribute_results(attribute: str) -> pd.DataFrame:
	file_path = resolve_results_path(attribute)
	df = pd.read_csv(file_path)

	required_columns = {"term", "ame"}
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
	return input_df[["term", "p_selected", "ame"]].sort_values("term")


def all_significant_same_direction(input_df: pd.DataFrame, alpha: float = 0.05) -> bool:
	if not (input_df["p_selected"] < alpha).all():
		return False

	ame_signs = {1 if value > 0 else -1 if value < 0 else 0 for value in input_df["ame"]}
	return len(ame_signs) == 1 and 0 not in ame_signs


def evaluate_attribute(input_df: pd.DataFrame, alpha: float = 0.05) -> tuple[bool, str | None]:
	failed_reasons: list[str] = []

	nonsignificant_terms = input_df.loc[input_df["p_selected"] >= alpha, "term"].tolist()
	if nonsignificant_terms:
		failed_reasons.append(
			"non-significant coefficient(s): " + ", ".join(nonsignificant_terms)
		)

	zero_ame_terms = input_df.loc[input_df["ame"] == 0, "term"].tolist()
	if zero_ame_terms:
		failed_reasons.append("zero AME(s): " + ", ".join(zero_ame_terms))

	nonzero_signs = {1 if value > 0 else -1 for value in input_df.loc[input_df["ame"] != 0, "ame"]}
	if len(nonzero_signs) > 1:
		ame_sign_summary = ", ".join(
			f"{row.term}={'positive' if row.ame > 0 else 'negative'}"
			for row in input_df.itertuples()
		)
		failed_reasons.append("mixed AME directions: " + ame_sign_summary)

	if failed_reasons:
		return False, "; ".join(failed_reasons)

	return True, None


def main() -> None:
	matching_attributes: list[str] = []
	failing_attributes: list[tuple[str, str]] = []
	all_results: list[pd.DataFrame] = []

	for attribute in SCALE_ATTRIBUTES:
		input_df = load_attribute_results(attribute)
		input_df["attribute"] = attribute
		all_results.append(input_df)
		matches, failure_reason = evaluate_attribute(input_df)
		if matches:
			matching_attributes.append(attribute)
		else:
			failing_attributes.append((attribute, failure_reason or "unknown reason"))

	average_absolute_ame_table = (
		pd.concat(all_results, ignore_index=True)
		.assign(
			input_condition=lambda df: df["term"].str.removeprefix("input_condition_"),
			absolute_ame=lambda df: df["ame"].abs(),
		)
		.groupby("input_condition", as_index=False)
		.agg(
			n_attributes=("attribute", "nunique"),
			average_absolute_ame=("absolute_ame", "mean"),
		)
		.sort_values("average_absolute_ame", ascending=False)
	)
	average_absolute_ame_table["average_absolute_ame"] = average_absolute_ame_table[
		"average_absolute_ame"
	].round(3)

	rank_count_table = (
		pd.concat(all_results, ignore_index=True)
		.assign(
			input_condition=lambda df: df["term"].str.removeprefix("input_condition_"),
			absolute_ame=lambda df: df["ame"].abs(),
		)
		.sort_values(["attribute", "absolute_ame", "input_condition"], ascending=[True, False, True])
		.assign(rank_within_attribute=lambda df: df.groupby("attribute").cumcount() + 1)
		.pivot_table(
			index="input_condition",
			columns="rank_within_attribute",
			values="attribute",
			aggfunc="count",
			fill_value=0,
		)
		.rename(columns={1: "largest", 2: "second_largest", 3: "third_largest", 4: "fourth_largest"})
		.reset_index()
	)

	print(
		f"Attributes where all four input-condition coefficients are significant and AMEs have the same sign: "
		f"{len(matching_attributes)} / {len(SCALE_ATTRIBUTES)}"
	)

	if matching_attributes:
		print("Matching attributes:")
		for attribute in matching_attributes:
			print(attribute)

	if failing_attributes:
		print("Failing attributes:")
		for attribute, reason in failing_attributes:
			print(f"{attribute}: {reason}")

	print("Average absolute AME by input condition across scale attributes:")
	print(average_absolute_ame_table.to_string(index=False))

	print("Rank counts by absolute AME within each scale attribute:")
	print(rank_count_table.to_string(index=False))


if __name__ == "__main__":
	main()