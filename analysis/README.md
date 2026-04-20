# Analysis

This directory contains the analysis scripts used for the main study and the two follow-up studies.
Each study has its own subdirectory, alongside shared overview and preprocessing scripts.

- Root-level scripts contain cross-study summaries and shared preprocessing steps.
- Study-specific subdirectories contain phase-specific statistical analyses and plotting scripts.

## Directory Overview

| Directory | Description |
|------|-------------|
| `.` | Cross-study summaries, shared preprocessing scripts, and study-specific analysis subdirectories. |
| `1_main_study` | Phase 1 and Phase 2 analyses for the main writer-preference and reader-annotation study. |
| `2_disclaimer_study` | Phase 1 analyses for the follow-up study testing disclaimer conditions. |
| `3_mitigation_study` | Phase 1 and Phase 2 analyses for the follow-up study testing distortion-mitigation strategies. |
| `utils_py` | Shared Python helpers for plotting offsets and variable group definitions. |
| `utils_r` | Shared R helpers for loading standard Phase 2 data splits and defining analysis variables. |

## Root-Level Scripts

Top-level analysis scripts and shared entry points.

| File | Description |
|------|-------------|
| `data_aggregation.py` | Aggregates Phase 2 annotation rows into paragraph-level summaries and writes `annotations_aggregated.csv` files. |
| `participant_overview.py` | Summarizes participant demographics and assignment coverage across all studies and writes overview tables. |
| `study_overview.R` | Prints cross-study participant counts, rating volume, and assignment structure summaries to the console. |

## Main Study

### `1_main_study`

Scripts for the main study analyses.

| File | Description |
|------|-------------|
| `phase_1_distortion_tolerance.py` | Summarizes writer-reported tolerance for AI-induced distortions and saves Phase 1 summary figures. |
| `phase_1_paragraph_edits.py` | Computes writer edit counts and Levenshtein-based edit metrics for model paragraphs. |
| `phase_1_paragraph_preference.R` | Analyzes writer preferences, edit behavior, and preference reasons for original versus model-edited paragraphs. |
| `phase_1_writer_engagement.py` | Visualizes writer self-reported issue knowledge, importance, confidence, and stance distributions. |

### Phase 2 Distributions

Scripts for the main study reader-side distribution analyses.

| File | Description |
|------|-------------|
| `phase_2_distribution_variables.R` | Runs the main-study Phase 2 distribution tests for scale, ordinal, and nominal variables and saves the shared result tables and scale correlation outputs. |
| `phase_2_distribution_plots.py` | Generates Phase 2 distribution figures comparing writer and model annotation distributions across data splits. |
| `phase_2_homogenisation.R` | Quantifies homogenisation by comparing spread and uncertainty across writer and model annotation distributions. |

### Phase 2 Distortions

Scripts for the main study reader-side distortion analyses.

| File | Description |
|------|-------------|
| `phase_2_distortion_by_input_condition.py` | Collates per-attribute distortion outputs into split-specific summaries by input condition. |
| `phase_2_distortion_by_model.py` | Collates per-attribute distortion outputs into split-specific summaries by model. |
| `phase_2_distortion_by_proposition_leaning.R` | Re-runs preferred-subset distortion models separately for left- and right-leaning propositions. |
| `phase_2_distortion_nominal_variables.R` | Estimates nominal distortion effects for writer versus model text with multinomial and one-vs-all models. |
| `phase_2_distortion_ordinal_variables.R` | Estimates ordinal distortion effects for writer versus model text with cumulative link mixed models. |
| `phase_2_distortion_plots.py` | Builds publication-style distortion figures across models, input conditions, and proposition-leaning subsets. |
| `phase_2_distortion_scale_variables.R` | Estimates scale-based distortion effects with beta regressions and average marginal effects. |

## Follow-Up Studies

### `2_disclaimer_study`

Scripts for the disclaimer-condition follow-up analyses.

| File | Description |
|------|-------------|
| `phase_1_distortion_tolerance.py` | Summarizes writer-reported distortion tolerance under disclaimer prompting conditions. |
| `phase_1_paragraph_preference.R` | Analyzes writer preferences and edit behavior for disclaimer-conditioned model paragraphs. |

### `3_mitigation_study`

Scripts for the mitigation-strategy follow-up analyses.

| File | Description |
|------|-------------|
| `phase_1_paragraph_preference.R` | Analyzes writer preferences and edit behavior across mitigation conditions and models. |
| `phase_2_distortion_nominal_variables.R` | Estimates nominal distortion effects for mitigation-conditioned model output. |
| `phase_2_distortion_ordinal_variables.R` | Estimates ordinal distortion effects for mitigation-conditioned model output. |
| `phase_2_distortion_plots.py` | Generates Phase 2 distortion figures comparing mitigation conditions across outcomes and splits. |
| `phase_2_distortion_scale_variables.R` | Estimates scale-based distortion effects for mitigation-conditioned model output. |
| `phase_2_distortion_side_effects.R` | Summarizes preferred-subset mitigation side effects from the distortion regression outputs. |
| `phase_2_distribution_plots.py` | Generates Phase 2 distribution figures for writer versus mitigation-conditioned model annotations. |
| `phase_2_distribution_variables.R` | Runs the mitigation-study Phase 2 distribution tests for scale, ordinal, and nominal variables and saves the shared result tables and scale correlation outputs. |
| `phase_2_mitigation_side_effects_plots.py` | Visualizes trade-offs between mitigation gains and side-effect outcomes. |

## Shared Utilities

### `utils_py`

Shared Python helpers used by multiple plotting and summary scripts.

| File | Description |
|------|-------------|
| `plotting_utils.py` | Provides helper functions for positioning grouped plot elements. |
| `variable_definitions.py` | Defines shared attribute lists, factor levels, and comparison term groupings for Python scripts. |

### `utils_r`

Shared R helpers used by multiple Phase 2 analysis scripts.

| File | Description |
|------|-------------|
| `data_loading.R` | Loads Phase 2 annotations and constructs the standard unedited, edited, and preferred data splits. |
| `variable_definitions.R` | Defines shared rating-variable lists and ordered factor levels for R analyses. |