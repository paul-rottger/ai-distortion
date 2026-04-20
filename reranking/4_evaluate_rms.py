#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FINETUNING_DATA_DIR = SCRIPT_DIR / "finetuning_data"
DEFAULT_SCORE_ROOT = SCRIPT_DIR / "rm_scores"
OUTPUT_ROOT = SCRIPT_DIR / "rm_evaluations"
DATASET_CHOICES = ("bullet_rm", "paragraph_rm")
JOIN_COLUMNS = [
    "writer_id",
    "proposition_id",
    "paragraph_type",
    "model_name",
    "model_input_condition",
]
OPTIONAL_JOIN_COLUMNS = ["model_name", "model_input_condition"]
TARGET_COLUMN = "writer_stance"
PREDICTION_COLUMN = "pred_writer_stance"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reward-model score files against the prepared test set."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="paragraph_rm",
        help="Dataset under reranking/finetuning_data to evaluate against.",
    )
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=None,
        help="Optional explicit score CSV files to evaluate. Defaults to all CSVs in the dataset score directory.",
    )
    parser.add_argument(
        "--score-root",
        default=str(DEFAULT_SCORE_ROOT),
        help="Root directory containing per-dataset score CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and planned evaluations without reading score files in depth or writing outputs.",
    )
    return parser.parse_args()


def model_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def dataset_test_path(dataset_name: str) -> Path:
    return FINETUNING_DATA_DIR / dataset_name / "test.csv"


def normalize_join_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in OPTIONAL_JOIN_COLUMNS:
        normalized[column] = normalized[column].fillna("")
    return normalized


def load_test_df(dataset_name: str) -> pd.DataFrame:
    path = dataset_test_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing test data: {path}")
    return normalize_join_columns(pd.read_csv(path))


def resolve_score_files(dataset_name: str, score_root: Path, explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        score_files = [Path(path).expanduser().resolve() for path in explicit_paths]
    else:
        dataset_dir = score_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Score directory does not exist: {dataset_dir}")
        score_files = sorted(dataset_dir.glob("*.csv"))

    if not score_files:
        raise FileNotFoundError("No score files found to evaluate.")

    missing = [path for path in score_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")

    return score_files

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return math.sqrt(float(((y_true - y_pred) ** 2).mean()))


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def accuracy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true == y_pred).mean())


def macro_f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1_values: list[float] = []

    for label in labels:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0

        if precision == 0.0 and recall == 0.0:
            f1_values.append(0.0)
            continue

        f1_values.append((2 * precision * recall) / (precision + recall))

    return float(sum(f1_values) / len(f1_values)) if f1_values else float("nan")


def load_score_df(score_path: Path) -> pd.DataFrame:
    score_df = pd.read_csv(score_path)
    if PREDICTION_COLUMN not in score_df.columns:
        raise ValueError(f"Missing prediction column {PREDICTION_COLUMN} in {score_path}")
    return normalize_join_columns(score_df)


def merge_scores(test_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    score_columns = JOIN_COLUMNS + [PREDICTION_COLUMN]
    if "eval_model" in score_df.columns:
        score_columns.append("eval_model")
    if "eval_text" in score_df.columns:
        score_columns.append("eval_text")

    merged = test_df.merge(
        score_df[score_columns],
        on=JOIN_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    return merged


def evaluate_predictions(merged_df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    matched_df = merged_df.dropna(subset=[PREDICTION_COLUMN]).copy()
    if matched_df.empty:
        raise ValueError(f"No scored rows available for {label}")

    y_true = matched_df[TARGET_COLUMN].astype(float)
    y_pred = matched_df[PREDICTION_COLUMN].astype(float)

    return {
        "model": label,
        "n_test_rows": int(len(merged_df)),
        "n_scored_rows": int(len(matched_df)),
        "n_missing_rows": int(merged_df[PREDICTION_COLUMN].isna().sum()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }


def main() -> None:
    args = parse_args()

    score_root = Path(args.score_root).expanduser().resolve()
    test_df = load_test_df(args.dataset)
    output_dir = OUTPUT_ROOT
    default_score_dir = score_root / args.dataset

    if args.dry_run:
        try:
            score_files = resolve_score_files(args.dataset, score_root, args.score_files)
        except FileNotFoundError:
            score_files = []
    else:
        score_files = resolve_score_files(args.dataset, score_root, args.score_files)

    print(f"Dataset: {args.dataset}")
    print(f"Test file: {dataset_test_path(args.dataset)}")
    print(f"Score root: {score_root}")
    print(f"Expected dataset score dir: {default_score_dir}")
    print(f"Score files: {[str(path) for path in score_files]}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        return

    summaries: list[dict[str, float | int | str]] = []

    for score_path in score_files:
        label = score_path.stem
        score_df = load_score_df(score_path)
        merged_df = merge_scores(test_df, score_df)
        summary = evaluate_predictions(merged_df, label)
        summaries.append(summary)
        print(
            f"Evaluated {label}: n={summary['n_scored_rows']}, "
            f"MAE={summary['mae']:.3f}, RMSE={summary['rmse']:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values("mae", ascending=True)
    summary_path = output_dir / f"{args.dataset}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
