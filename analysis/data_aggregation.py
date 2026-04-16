#!/usr/bin/env python3

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Aggregation configuration
GROUP_COLUMNS = ["writer_id", "proposition_id", "paragraph_type"]
EXCLUDED_COLUMNS = {"rater_id"}
SOURCE_FILE_NAME = "annotations.csv"
OUTPUT_FILE_NAME = "annotations_aggregated.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

def phase_2_annotation_paths() -> list[Path]:
	return sorted(
		path
		for path in DATA_DIR.glob(f"**/{SOURCE_FILE_NAME}")
		if "phase_2" in path.parent.name
	)


# =============================================================================
# ANALYSIS
# =============================================================================

def modal_value(series: pd.Series):
	non_null_values = [value for value in series.tolist() if pd.notna(value)]
	if not non_null_values:
		return pd.NA

	counts: dict[object, int] = {}
	for value in non_null_values:
		counts[value] = counts.get(value, 0) + 1

	max_count = max(counts.values())
	for value in non_null_values:
		if counts[value] == max_count:
			return value

	return pd.NA


def aggregate_annotations(df: pd.DataFrame) -> pd.DataFrame:
	missing_columns = [column for column in GROUP_COLUMNS if column not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing required grouping columns: {missing_columns}")

	aggregation_map: dict[str, str] = {}
	for column in df.columns:
		if column in GROUP_COLUMNS or column in EXCLUDED_COLUMNS:
			continue

		aggregation_map[column] = "mean" if is_numeric_dtype(df[column]) else modal_value

	aggregated_df = (
		df.groupby(GROUP_COLUMNS, dropna=False, sort=False)
		.agg(aggregation_map)
		.reset_index()
	)

	output_columns = [
		column for column in df.columns if column not in EXCLUDED_COLUMNS
	]
	return aggregated_df[output_columns]


def aggregate_file(source_path: Path) -> Path:
	df = pd.read_csv(source_path)
	aggregated_df = aggregate_annotations(df)
	output_path = source_path.with_name(OUTPUT_FILE_NAME)
	aggregated_df.to_csv(output_path, index=False)
	return output_path


# =============================================================================
# OUTPUTS
# =============================================================================

def main() -> None:
	annotation_paths = phase_2_annotation_paths()
	if not annotation_paths:
		raise FileNotFoundError("No phase 2 annotations.csv files found under data/.")

	for source_path in annotation_paths:
		output_path = aggregate_file(source_path)
		print(f"Wrote {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
	main()
