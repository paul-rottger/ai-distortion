#!/usr/bin/env python3

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path

import pandas as pd

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CENSUS_PATH = DATA_DIR / "ext" / "uk_census_2021.csv"
RESULTS_DIR = BASE_DIR / "results" / "participant_demographics"
ASSIGNMENT_RESULTS_DIR = BASE_DIR / "results" / "participant_assignment"

# Study configuration
STUDIES = [
    "main_phase_1",
    "main_phase_2",
    "followup_disclaimer_phase_1",
    "followup_mitigation_phase_1",
    "followup_mitigation_phase_2",
]

PHASE_2_STUDIES = [study for study in STUDIES if "phase_2" in study]
PARAGRAPH_ID_COLUMNS = ["writer_id", "proposition_id", "paragraph_type"]

PARTICIPANT_ID_COLUMNS = ["writer_id", "rater_id"]

CENSUS_ATTRIBUTES = {"age_binned", "gender", "race"}

ATTRIBUTE_CONFIG = {
    "age_binned": ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    "gender": ["Female", "Male", "Other", "Prefer not to say"],
    "race": ["White", "Black", "Asian", "Mixed", "Other", "Prefer not to say"],
    "englishFirst": ["Yes", "No", "Bilingual from birth", "Prefer not to say"],
    "englishSkills": [
        "Basic",
        "Intermediate",
        "Advanced",
        "Expert",
        "Prefer not to say",
    ],
    "education": [
        "GCSEs or equivalent",
        "A-levels or equivalent",
        "Vocational qualification",
        "Undergraduate degree",
        "Postgraduate degree (Master's)",
        "Doctorate (PhD)",
        "Other",
        "Prefer not to say",
    ],
    "income": [
        "Under £15,000",
        "£15,000-£24,999",
        "£25,000-£34,999",
        "£35,000-£49,999",
        "£50,000-£74,999",
        "£75,000-£99,999",
        "£100,000+",
        "Prefer not to say",
    ],
    "politicalParty": [
        "Labour",
        "Conservative",
        "Liberal Democrats",
        "Greens",
        "Scottish National Party",
        "Reform UK",
        "Other",
        "Did not vote",
        "Not eligible to vote",
        "Prefer not to say",
    ],
    "politicalIdeology": [
        "Very Left-Wing",
        "Moderately Left-Wing",
        "Centrist",
        "Moderately Right-Wing",
        "Very Right-Wing",
        "Prefer not to say",
    ],
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_census_summaries() -> dict[str, pd.Series]:
    census_df = pd.read_csv(CENSUS_PATH)
    race_columns = ["Asian", "Black", "Mixed", "White", "Other"]

    for column in race_columns:
        census_df[column] = (
            census_df[column].astype(str).str.replace(",", "", regex=False).astype(int)
        )

    gender_summary = census_df.groupby("gender")[race_columns].sum().sum(axis=1)

    age_summary = (
        census_df.assign(age_binned=census_df["age"].map(census_age_to_binned))
        .groupby("age_binned")[race_columns]
        .sum()
        .sum(axis=1)
        .reindex(ATTRIBUTE_CONFIG["age_binned"], fill_value=0)
    )

    race_summary = census_df[race_columns].sum().reindex(race_columns, fill_value=0)

    return {
        "gender": gender_summary.reindex(["Female", "Male"], fill_value=0),
        "age_binned": age_summary,
        "race": race_summary,
    }


def load_participants_by_study() -> dict[str, pd.DataFrame]:
    participants_by_study: dict[str, pd.DataFrame] = {}

    for study in STUDIES:
        participant_path = DATA_DIR / study / "participants.csv"
        df = pd.read_csv(participant_path)

        participant_id_column = next(
            (column for column in PARTICIPANT_ID_COLUMNS if column in df.columns),
            None,
        )
        if participant_id_column is None:
            raise ValueError(f"No participant ID column found for {study}")

        df = df.rename(columns={participant_id_column: "participant_id"})

        df["age_binned"] = pd.cut(
            df["age"],
            bins=[18, 30, 40, 50, 60, 70, float("inf")],
            right=False,
            labels=ATTRIBUTE_CONFIG["age_binned"],
        ).astype("object")

        participants_by_study[study] = df

    return participants_by_study


def load_phase_2_annotations_by_study() -> dict[str, pd.DataFrame]:
    annotations_by_study: dict[str, pd.DataFrame] = {}

    for study in PHASE_2_STUDIES:
        annotations_path = DATA_DIR / study / "annotations.csv"
        annotations_by_study[study] = pd.read_csv(annotations_path)

    return annotations_by_study


# =============================================================================
# ANALYSIS
# =============================================================================

def format_count_percent(counts: pd.Series, total_n: int) -> list[str]:
    percents = (counts / total_n) * 100
    return [
        f"{count} ({percent:.1f}%)"
        for count, percent in zip(counts.tolist(), percents.tolist())
    ]


def census_age_to_binned(age_label: str) -> str:
    age_mapping = {
        "18-19": "18-29",
        "20-24": "18-29",
        "25-29": "18-29",
        "30-34": "30-39",
        "35-39": "30-39",
        "40-44": "40-49",
        "45-49": "40-49",
        "50-54": "50-59",
        "55-59": "50-59",
        "60-64": "60-69",
        "65-69": "60-69",
        "70-74": "70+",
        "75-79": "70+",
        "80-84": "70+",
        "85+": "70+",
    }
    return age_mapping[age_label]


def ordered_categories(series: pd.Series, attribute: str) -> list[str]:
    configured_order = ATTRIBUTE_CONFIG[attribute]
    observed_categories = {
        str(value)
        for value in series.fillna("Missing").astype(str).tolist()
        if str(value) != "nan"
    }

    categories = [category for category in configured_order if category in observed_categories]
    remaining = sorted(observed_categories - set(configured_order) - {"Missing"})
    categories.extend(remaining)

    if "Missing" in observed_categories:
        categories.append("Missing")

    return categories


def deduplicate_participants(
    participants_by_study: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined_participants = pd.concat(
        [
            df.assign(_study_order=study_index)
            for study_index, df in enumerate(participants_by_study.values())
        ],
        ignore_index=True,
    )

    combined_participants = combined_participants.sort_values(
        ["participant_id", "_study_order"],
        kind="stable",
    )

    return combined_participants.drop_duplicates(
        subset="participant_id",
        keep="first",
    ).drop(columns="_study_order")


def summarize_attribute_wide(
    participants_by_study: dict[str, pd.DataFrame],
    attribute: str,
    census_summaries: dict[str, pd.Series],
) -> pd.DataFrame:
    combined_values = pd.concat(
        [df[attribute] for df in participants_by_study.values()],
        ignore_index=True,
    )
    categories = ordered_categories(combined_values, attribute)

    summary_df = pd.DataFrame({attribute: categories})

    for study, df in participants_by_study.items():
        counts = (
            df[attribute]
            .fillna("Missing")
            .astype(str)
            .value_counts(dropna=False)
            .reindex(categories, fill_value=0)
        )
        total_n = len(df)
        summary_df[study] = format_count_percent(counts, total_n)

    if attribute in CENSUS_ATTRIBUTES:
        census_counts = census_summaries[attribute].reindex(categories, fill_value=0)
        summary_df["census"] = format_count_percent(census_counts, int(census_counts.sum()))

    deduplicated_participants = deduplicate_participants(participants_by_study)
    total_counts = (
        deduplicated_participants[attribute]
        .fillna("Missing")
        .astype(str)
        .value_counts(dropna=False)
        .reindex(categories, fill_value=0)
    )
    summary_df["total"] = format_count_percent(total_counts, len(deduplicated_participants))

    total_row = {attribute: "n_participants"}
    for study, df in participants_by_study.items():
        total_row[study] = str(len(df))

    if attribute in CENSUS_ATTRIBUTES:
        total_row["census"] = str(int(census_summaries[attribute].sum()))

    total_row["total"] = str(len(deduplicated_participants))

    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    return summary_df


def participant_ids_by_study(
    participants_by_study: dict[str, pd.DataFrame],
) -> dict[str, set[str]]:
    return {
        study: set(df["participant_id"].dropna().astype(str))
        for study, df in participants_by_study.items()
    }


def print_participant_overlap(participants_by_study: dict[str, pd.DataFrame]) -> None:
    participant_ids = participant_ids_by_study(participants_by_study)

    print("\n=== participant_overlap ===")

    has_overlap = False
    for index, study_a in enumerate(STUDIES):
        ids_a = participant_ids[study_a]
        for study_b in STUDIES[index + 1 :]:
            ids_b = participant_ids[study_b]
            overlap = ids_a & ids_b
            overlap_count = len(overlap)

            pct_a = (overlap_count / len(ids_a) * 100) if ids_a else 0.0
            pct_b = (overlap_count / len(ids_b) * 100) if ids_b else 0.0

            print(
                f"{study_a} vs {study_b}: {overlap_count} overlapping participants "
                f"({pct_a:.1f}% of {study_a}; {pct_b:.1f}% of {study_b})"
            )

            if overlap_count:
                has_overlap = True

    if not has_overlap:
        print("No participant overlap detected across studies.")


def write_attribute_summaries(participants_by_study: dict[str, pd.DataFrame]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    census_summaries = load_census_summaries()

    for attribute in ATTRIBUTE_CONFIG:
        summary_df = summarize_attribute_wide(
            participants_by_study,
            attribute,
            census_summaries,
        )
        summary_df.to_csv(RESULTS_DIR / f"{attribute}.csv", index=False)
        print(f"\n=== {attribute} ===")
        print(summary_df.to_latex(index=False, escape=False))


def summarize_count_distribution(
    counts: pd.Series,
    frequency_column: str,
    proportion_column: str,
) -> pd.DataFrame:
    distribution = counts.value_counts().sort_index()

    return pd.DataFrame(
        {
            counts.name: distribution.index.astype(int),
            frequency_column: distribution.astype(int).to_numpy(),
            proportion_column: (distribution / len(counts)).round(4).to_numpy(),
        }
    )


def write_assignment_summaries(annotations_by_study: dict[str, pd.DataFrame]) -> None:
    ASSIGNMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for study, df in annotations_by_study.items():
        rater_counts = df.groupby("rater_id").size().rename("paragraphs_rated")
        paragraph_counts = (
            df.groupby(PARAGRAPH_ID_COLUMNS).size().rename("ratings_received")
        )

        rater_summary = summarize_count_distribution(
            rater_counts,
            frequency_column="n_raters",
            proportion_column="proportion_raters",
        )
        paragraph_summary = summarize_count_distribution(
            paragraph_counts,
            frequency_column="n_paragraphs",
            proportion_column="proportion_paragraphs",
        )

        rater_summary.to_csv(
            ASSIGNMENT_RESULTS_DIR / f"{study}_raters_by_paragraph_count.csv",
            index=False,
        )
        paragraph_summary.to_csv(
            ASSIGNMENT_RESULTS_DIR / f"{study}_paragraphs_by_rating_count.csv",
            index=False,
        )


# =============================================================================
# OUTPUTS
# =============================================================================

def main() -> None:
    participants_by_study = load_participants_by_study()
    annotations_by_study = load_phase_2_annotations_by_study()
    write_attribute_summaries(participants_by_study)
    write_assignment_summaries(annotations_by_study)
    print_participant_overlap(participants_by_study)


if __name__ == "__main__":
    main()