from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "participant_demographics"

STUDIES = [
    "main_phase_1",
    "main_phase_2",
    "followup_disclaimer_phase_1",
    "followup_mitigation_phase_1",
    "followup_mitigation_phase_2",
]

PARTICIPANT_ID_COLUMNS = ["writer_id", "rater_id"]

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


def load_participants_by_study() -> dict[str, pd.DataFrame]:
    participants_by_study: dict[str, pd.DataFrame] = {}

    for study in STUDIES:
        participant_path = DATA_DIR / study / "participants.csv"
        df = pd.read_csv(participant_path)

        participant_id_column = next(
            (column for column in PARTICIPANT_ID_COLUMNS if column in df.columns),
            None,
        )
        if participant_id_column is not None:
            df = df.rename(columns={participant_id_column: "participant_id"})

        df["age_binned"] = pd.cut(
            df["age"],
            bins=[18, 30, 40, 50, 60, 70, float("inf")],
            right=False,
            labels=ATTRIBUTE_CONFIG["age_binned"],
        ).astype("object")

        participants_by_study[study] = df

    return participants_by_study


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


def summarize_attribute_wide(
    participants_by_study: dict[str, pd.DataFrame], attribute: str
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
        percents = (counts / total_n) * 100
        summary_df[study] = [
            f"{count} ({percent:.1f}%)"
            for count, percent in zip(counts.tolist(), percents.tolist())
        ]

    total_row = {attribute: "n_participants"}
    for study, df in participants_by_study.items():
        total_row[study] = str(len(df))

    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    return summary_df


def write_attribute_summaries(participants_by_study: dict[str, pd.DataFrame]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for attribute in ATTRIBUTE_CONFIG:
        summary_df = summarize_attribute_wide(participants_by_study, attribute)
        summary_df.to_csv(RESULTS_DIR / f"{attribute}.csv", index=False)
        print(f"\n=== {attribute} ===")
        print(summary_df.to_latex(index=False, escape=False))


def main() -> None:
    participants_by_study = load_participants_by_study()
    write_attribute_summaries(participants_by_study)


if __name__ == "__main__":
    main()