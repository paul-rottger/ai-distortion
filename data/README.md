# Data

This directory contains the data used in the main study and the two follow-up studies (disclaimer and mitigation).
Each study-phase combination has its own subdirectory.

- Phase 1 contains writer-side data: participant demographics, propositions, written paragraphs, AI-edited paragraphs, preferences, and self-reported distortion tolerance.
- Phase 2 contains reader-side annotation data: participant demographics and paragraph ratings.
- Files ending in `_summary.csv` or `_aggregated.csv` are derived analysis inputs generated from collected data.

## Directory Overview

| Directory | Description |
|------|-------------|
| `main_phase_1` | Main study, writer phase |
| `main_phase_2` | Main study, reader annotation phase |
| `followup_disclaimer_phase_1` | Disclaimer follow-up, writer phase |
| `followup_mitigation_phase_1` | Mitigation follow-up, writer phase |
| `followup_mitigation_phase_2` | Mitigation follow-up, reader annotation phase |
| `ext` | External reference data used in analysis |

## Main Study

### `main_phase_1`

Collected writer-side data and derived summaries for the main study.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 1,501 | Writer participant demographics and completion metadata |
| `propositions.csv` | 100 | Proposition pool used for writing tasks |
| `proposition_responses.csv` | 16,565 | Per-proposition writer responses, including stance, bullets, original paragraph, model outputs, edited paragraph, and preference fields |
| `paragraphs.csv` | 13,444 | Long-format paragraph table with one row per paragraph variant (`writer` or AI-generated) |
| `distortion_responses.csv` | 1,501 | Writer ratings of how AI editing could change perceived stance, style, affect, demographics, and politics |
| `distortion_responses_summary.csv` | 49 | Derived summary statistics for distortion tolerance items, including bootstrap confidence intervals |
| `distortion_responses_binned_summary.csv` | 49 | Derived binned summary of distortion tolerance items for plotting/reporting |

### `main_phase_2`

Collected reader-side annotation data for the main study.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 10,017 | Reader participant demographics and completion metadata |
| `annotations.csv` | 100,124 | Raw paragraph-level annotations from readers, with one row per rater-paragraph judgment |
| `annotations_aggregated.csv` | 10,008 | Derived paragraph-level aggregation of `annotations.csv`, averaging numeric measures and taking the modal categorical value |

## Follow-Up Studies

### `followup_disclaimer_phase_1`

Writer-side data for the disclaimer-condition follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 669 | Writer participant demographics and completion metadata |
| `proposition_responses.csv` | 7,483 | Per-proposition writer responses, including disclaimer condition, model output, edited paragraph, and preference fields |
| `distortion_responses.csv` | 669 | Writer ratings for the disclaimer follow-up distortion tolerance items |
| `distortion_responses_summary.csv` | 8 | Derived summary statistics for the disclaimer follow-up distortion tolerance measures |

### `followup_mitigation_phase_1`

Writer-side data for the mitigation-strategy follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 769 | Writer participant demographics and completion metadata |
| `proposition_responses.csv` | 7,908 | Per-proposition writer responses, including mitigation-related model conditions, edited paragraph, and preference fields |
| `paragraphs.csv` | 6,267 | Long-format paragraph table with one row per writer or AI-generated paragraph variant |

### `followup_mitigation_phase_2`

Reader-side annotation data for the mitigation-strategy follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 2,543 | Reader participant demographics and completion metadata |
| `annotations.csv` | 25,422 | Raw paragraph-level annotations, including mitigation condition metadata |
| `annotations_aggregated.csv` | 5,016 | Derived paragraph-level aggregation of `annotations.csv` |

## External Data

### `ext`

Reference datasets used in downstream analyses but not collected in the study itself.

| File | Rows | Description |
|------|------|-------------|
| `uk_census_2021.csv` | 29 | UK Census 2021 reference table from the [ONS website](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/censusbasedstatisticsuk2021) |

## Notes

- Phase 1 participant files use `writer_id`; Phase 2 participant files use `rater_id`.
- `annotations_aggregated.csv` files are produced by aggregating `annotations.csv` by writer, proposition, and paragraph type.
- `distortion_responses_summary.csv` files are produced from `distortion_responses.csv` and are used by the distortion tolerance plotting scripts.