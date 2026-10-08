# Data

This directory contains the data used in the main study and the four follow-up studies (disclaimer, mitigation, trust, and persuasion).
Each study-phase combination has its own subdirectory.

- Phase 1 contains writer-side data: participant demographics, propositions, written paragraphs, AI-edited paragraphs, preferences, and self-reported distortion tolerance.
- Phase 2 contains reader-side annotation data: participant demographics and paragraph ratings.
- The trust and persuasion follow-ups collected only reader-side data and so are not split into phases.
- Files ending in `_summary.csv` or `_aggregated.csv` are derived analysis inputs generated from collected data.

## Directory Overview

| Directory | Description |
|------|-------------|
| `main_phase_1` | Main study (Study 1), writer phase |
| `main_phase_2` | Main study (Study 1), reader annotation phase |
| `followup_disclaimer_phase_1` | Disclaimer follow-up (Study 2), writer phase |
| `followup_mitigation_phase_1` | Mitigation follow-up (Study 3), writer phase |
| `followup_mitigation_phase_2` | Mitigation follow-up (Study 3), reader annotation phase |
| `followup_trust` | Trust follow-up (Study 4), reader phase |
| `followup_persuasion` | Persuasion follow-up (Study 5), reader phase |
| `ext` | External reference data used in analysis |

## Main Study

### `main_phase_1`

Collected writer-side data and derived summaries for the main study.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 1,501 | Writer demographics and completion metadata |
| `propositions.csv` | 100 | Proposition pool used for writing tasks |
| `proposition_responses.csv` | 4,503 | Writer responses per proposition, including bullets, paragraphs, and preferences |
| `paragraphs.csv` | 10,008 | One row per writer or AI-generated paragraph |
| `distortion_responses.csv` | 1,501 | Writer tolerance for AI-induced distortions |
| `distortion_responses_summary.csv` | 49 | Summary statistics for `distortion_responses.csv` |

### `main_phase_2`

Collected reader-side annotation data for the main study.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 10,017 | Reader demographics and completion metadata |
| `annotations.csv` | 100,124 | Reader annotations, with one row per rater-paragraph judgment |
| `annotations_aggregated.csv` | 10,008 | Paragraph-level aggregation of `annotations.csv` |

## Follow-Up Studies

### `followup_disclaimer_phase_1`

Writer-side data for the disclaimer-condition follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 669 | Writer demographics and completion metadata |
| `proposition_responses.csv` | 2,007 | Writer responses per proposition, including disclaimer condition and preferences |
| `distortion_responses.csv` | 669 | Writer tolerance for AI-induced distortions |
| `distortion_responses_summary.csv` | 8 | Summary statistics for `distortion_responses.csv` |

### `followup_mitigation_phase_1`

Writer-side data for the mitigation-strategy follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 769 | Writer demographics and completion metadata |
| `proposition_responses.csv` | 2,307 | Writer responses per proposition, including mitigation condition and preferences |
| `paragraphs.csv` | 5,016 | One row per writer or AI-generated paragraph |

### `followup_mitigation_phase_2`

Reader-side annotation data for the mitigation-strategy follow-up.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 2,543 | Reader demographics and completion metadata |
| `annotations.csv` | 25,422 | Reader annotations, with one row per rater-paragraph judgment |
| `annotations_aggregated.csv` | 5,016 | Paragraph-level aggregation of `annotations.csv` |

### `followup_trust`

Reader-side data for the trust follow-up, in which readers allocated money to paragraph authors in a Trust Game.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 802 | Reader demographics and completion metadata |
| `annotations.csv` | 3,208 | Reader Trust Game allocations, with one row per rater-paragraph judgment |
| `paragraph_pairs.csv` | 3,795 | Main-study ratings of the writer and AI paragraph for each writer-proposition pair |

### `followup_persuasion`

Reader-side data for the persuasion follow-up, in which readers reported their stance before and after reading a paragraph.

| File | Rows | Description |
|------|------|-------------|
| `participants.csv` | 7,996 | Reader demographics and completion metadata |
| `annotations.csv` | 39,980 | Reader stance before and after reading, with one row per rater-paragraph judgment |
| `paragraph_pairs.csv` | 3,795 | Main-study ratings of the writer and AI paragraph for each writer-proposition pair |

## External Data

### `ext`

Reference datasets used in downstream analyses but not collected in the study itself.

| File | Rows | Description |
|------|------|-------------|
| `uk_census_2021.csv` | 30 | UK Census 2021 reference table from the [ONS website](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/censusbasedstatisticsuk2021) |

## Notes

- Phase 1 participant files use `writer_id`; Phase 2 participant files use `rater_id`.
- `annotations_aggregated.csv` files are produced by aggregating `annotations.csv` by writer, proposition, and paragraph type.
- `distortion_responses_summary.csv` files are produced from `distortion_responses.csv` and are used by the distortion tolerance plotting scripts.