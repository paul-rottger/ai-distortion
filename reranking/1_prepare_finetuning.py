#!/usr/bin/env python3

import json
import random
from numbers import Real
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
TRAIN_FRACTION = 0.75
VALID_FRACTION = 0.05
TEST_FRACTION = 0.20
TARGET_COLUMNS = ("writer_stance",)
TRAIN_SAMPLE_SIZES = (100, 500, 1000)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "finetuning_data"

ANNOTATIONS_PATH = DATA_DIR / "main_phase_2" / "annotations_aggregated.csv"
PARAGRAPHS_PATH = DATA_DIR / "main_phase_1" / "paragraphs.csv"
RESPONSES_PATH = DATA_DIR / "main_phase_1" / "proposition_responses.csv"

JOIN_COLUMNS = [
    "writer_id",
    "proposition_id",
    "paragraph_type",
    "model_name",
    "model_input_condition",
]
OPTIONAL_JOIN_COLUMNS = ["model_name", "model_input_condition"]

PARAGRAPH_PROMPT_TEMPLATE = """You will be shown an OPINION PARAGRAPH concerning a PROPOSITION.
We showed this paragraph to 10 independent raters and asked a RATING QUESTION.
For this rating question, your task is to predict the MEAN rating value across all raters.

RATING QUESTION:
writer_stance: What is the author's level of agreement with the proposition? [0-100 scale, where 0 = "Strongly disagree", 50 = "Ambivalent", and 100 = "Strongly agree"]

PROPOSITION: {proposition}

OPINION PARAGRAPH: {paragraph}
"""

BULLET_PROMPT_TEMPLATE = """You will be shown BULLET POINTS written by a person to explain their position on a PROPOSITION.
The person who wrote these bullet points expanded them into a full paragraph.
We showed this paragraph to 10 independent raters and asked a RATING QUESTION.
For this rating question, your task is to predict the MEAN rating value across all raters.

RATING QUESTION:
writer_stance: What is the author's level of agreement with the proposition? [0-100 scale, where 0 = "Strongly disagree", 50 = "Ambivalent", and 100 = "Strongly agree"]

PROPOSITION: {proposition}

BULLET POINTS: {writer_bullets}
"""


def validate_split_fractions() -> None:
    total_fraction = TRAIN_FRACTION + VALID_FRACTION + TEST_FRACTION
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test fractions must sum to 1.0. "
            f"Received {TRAIN_FRACTION=}, {VALID_FRACTION=}, {TEST_FRACTION=}."
        )

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_join_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in OPTIONAL_JOIN_COLUMNS:
        normalized[column] = normalized[column].fillna("")
    return normalized

def build_paragraph_dataset() -> pd.DataFrame:
    annotations_df = normalize_join_columns(load_csv(ANNOTATIONS_PATH))
    paragraphs_df = normalize_join_columns(load_csv(PARAGRAPHS_PATH))

    paragraph_df = annotations_df.merge(
        paragraphs_df[JOIN_COLUMNS + ["proposition", "paragraph"]],
        on=JOIN_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if paragraph_df["paragraph"].isna().any() or paragraph_df["proposition"].isna().any():
        raise ValueError("Failed to join annotation labels with paragraph text for all rows.")

    return paragraph_df.loc[paragraph_df["paragraph_type"] != "edited"].copy()

def build_bullet_dataset(paragraph_df: pd.DataFrame) -> pd.DataFrame:
    responses_df = load_csv(RESPONSES_PATH)
    writer_df = paragraph_df.loc[paragraph_df["paragraph_type"] == "writer"].copy()

    bullet_df = writer_df.merge(
        responses_df[["writer_id", "proposition_id", "writer_bullets"]],
        on=["writer_id", "proposition_id"],
        how="left",
        validate="one_to_one",
    )

    if bullet_df["writer_bullets"].isna().any():
        raise ValueError("Failed to join writer bullets for all bullet RM rows.")

    return bullet_df

def split_writer_ids(writer_ids: list[str]) -> dict[str, set[str]]:
    shuffled_writer_ids = writer_ids.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_writer_ids)

    n_writers = len(shuffled_writer_ids)
    n_test = int(TEST_FRACTION * n_writers)
    n_valid = int(VALID_FRACTION * n_writers)

    return {
        "test": set(shuffled_writer_ids[:n_test]),
        "valid": set(shuffled_writer_ids[n_test : n_test + n_valid]),
        "train": set(shuffled_writer_ids[n_test + n_valid :]),
    }

def split_dataframe_by_writer(df: pd.DataFrame, writer_splits: dict[str, set[str]]) -> dict[str, pd.DataFrame]:
    split_dfs: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        split_df = df.loc[df["writer_id"].isin(writer_splits[split_name])].copy()
        split_dfs[split_name] = split_df
    return split_dfs

def format_target_value(value: object) -> float | None | object:
    if pd.isna(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        return round(float(value), 1)
    return value

def row_to_record(row: pd.Series, prompt_template: str, prompt_fields: list[str]) -> dict[str, list[dict[str, str]]]:
    prompt_values = {field: row[field] for field in prompt_fields}
    user_prompt = prompt_template.format(**prompt_values)
    gold = {column: format_target_value(row[column]) for column in TARGET_COLUMNS}
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
        ]
    }

def write_jsonl(df: pd.DataFrame, output_path: Path, prompt_template: str, prompt_fields: list[str]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            record = row_to_record(row, prompt_template, prompt_fields)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_training_subsamples(
    train_df: pd.DataFrame,
    output_dir: Path,
    prompt_template: str,
    prompt_fields: list[str],
) -> None:
    full_train_path = output_dir / f"train_{len(train_df)}.jsonl"
    write_jsonl(train_df, full_train_path, prompt_template, prompt_fields)

    for sample_size in TRAIN_SAMPLE_SIZES:
        if len(train_df) < sample_size:
            continue
        sample_df = train_df.sample(sample_size, random_state=RANDOM_SEED + sample_size)
        write_jsonl(sample_df, output_dir / f"train_{sample_size}.jsonl", prompt_template, prompt_fields)

def export_dataset(
    dataset_name: str,
    dataset_df: pd.DataFrame,
    writer_splits: dict[str, set[str]],
    prompt_template: str,
    prompt_fields: list[str],
) -> None:
    output_dir = OUTPUT_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dfs = split_dataframe_by_writer(dataset_df, writer_splits)

    for split_name, split_df in split_dfs.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        write_jsonl(split_df, output_dir / f"{split_name}.jsonl", prompt_template, prompt_fields)

    write_training_subsamples(split_dfs["train"], output_dir, prompt_template, prompt_fields)

    print(f"{dataset_name}: {len(dataset_df)} rows")
    for split_name in ("train", "valid", "test"):
        split_df = split_dfs[split_name]
        print(
            f"  {split_name}: {len(split_df)} rows across {split_df['writer_id'].nunique()} writers"
        )

def main() -> None:
    validate_split_fractions()
    paragraph_df = build_paragraph_dataset()
    bullet_df = build_bullet_dataset(paragraph_df)

    writer_ids = paragraph_df["writer_id"].drop_duplicates().tolist()
    writer_splits = split_writer_ids(writer_ids)

    print(
        "Unique writers: "
        f"{len(writer_ids)} "
        f"(train: {len(writer_splits['train'])}, valid: {len(writer_splits['valid'])}, test: {len(writer_splits['test'])})"
    )

    export_dataset(
        dataset_name="paragraph_rm",
        dataset_df=paragraph_df,
        writer_splits=writer_splits,
        prompt_template=PARAGRAPH_PROMPT_TEMPLATE,
        prompt_fields=["proposition", "paragraph"],
    )
    export_dataset(
        dataset_name="bullet_rm",
        dataset_df=bullet_df,
        writer_splits=writer_splits,
        prompt_template=BULLET_PROMPT_TEMPLATE,
        prompt_fields=["proposition", "writer_bullets"],
    )

if __name__ == "__main__":
    main()
