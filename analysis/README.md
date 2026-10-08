# Analysis

This directory contains the analysis scripts used for the main study and the four follow-up studies.
Each study has its own subdirectory, alongside shared overview and preprocessing scripts.

- Root-level scripts contain cross-study summaries and shared preprocessing steps.
- Study-specific subdirectories contain phase-specific statistical analyses and plotting scripts.
- All scripts are run by `run_all.sh` in the repository root, except the preference-reason coding scripts described below.

## Preference Reasons

In Studies 1-3, writers explained in free text why they preferred one paragraph over the other.
We coded these free-text reasons with an LLM, using the codebook in `utils_files/preference_reason_codebook.json`, to enrich the collected data.
This coding does not need to be repeated, so the coding scripts (`phase_1_preference_reasons_coding.py`) in `1_main_study`, `2_disclaimer_study`, and `3_mitigation_study` are not part of `run_all.sh`.
The resulting codes are included in `results/<study>/preference_reasons/reason_codes.csv`.
The summary scripts (`phase_1_preference_reasons_summary.py`) run offline from these files and are part of `run_all.sh`.
Re-running the coding scripts requires an OpenRouter API key, set as `OPENROUTER_API_KEY` in a `.env` file in the repository root.

## Directory Overview

| Directory | Description |
|------|-------------|
| `.` | Cross-study and preprocessing scripts. |
| `1_main_study` | Main study (Study 1). |
| `2_disclaimer_study` | Disclaimer follow-up (Study 2). |
| `3_mitigation_study` | Mitigation follow-up (Study 3). |
| `4_trust_study` | Trust follow-up (Study 4). |
| `5_persuasion_study` | Persuasion follow-up (Study 5). |
| `utils_py` | Shared Python helpers. |
| `utils_r` | Shared R helpers. |
| `utils_files` | Shared non-code inputs. |

## Root-Level Scripts

Top-level analysis scripts and shared entry points.

| File | Description |
|------|-------------|
| `data_aggregation.py` | Aggregates Phase 2 annotations to paragraph level. |
| `participant_overview.py` | Summarizes participant demographics across all studies. |
| `study_overview.R` | Prints participant and rating counts across all studies. |
| `sensitivity_analysis.R` | Computes minimum detectable effects for headline contrasts in all studies (runs last, as it reads their results). |

## Main Study

### `1_main_study`

Scripts for the main study analyses.

| File | Description |
|------|-------------|
| `phase_1_distortion_tolerance.py` | Summarizes writer tolerance for AI-induced distortions. |
| `phase_1_paragraph_edits.py` | Measures how much writers edited model paragraphs. |
| `phase_1_text_metrics.py` | Compares text features and distinctive words of writer and model paragraphs. |
| `phase_1_text_homogenisation.py` | Compares the semantic and lexical similarity of writer versus model paragraphs. |
| `phase_1_paragraph_preference.R` | Analyzes writer preferences for their own versus model paragraphs. |
| `phase_1_preference_reasons_coding.py` | Codes free-text preference reasons with an LLM (see [Preference Reasons](#preference-reasons)). |
| `phase_1_preference_reasons_summary.py` | Summarizes the coded preference reasons. |
| `phase_1_writer_engagement.py` | Plots writers' self-reported issue engagement and stance. |

### Phase 2 Distributions

Scripts for the main study reader-side distribution analyses.

| File | Description |
|------|-------------|
| `phase_2_distribution_variables.R` | Tests for differences between writer and model annotation distributions. |
| `phase_2_distribution_plots.py` | Plots writer versus model annotation distributions. |
| `phase_2_homogenisation.R` | Tests whether model paragraphs are more homogeneous than writer paragraphs. |
| `phase_2_rater_agreement.R` | Computes inter-rater agreement for all annotation attributes. |

### Phase 2 Distortions

Scripts for the main study reader-side distortion analyses.

| File | Description |
|------|-------------|
| `phase_2_distortion_by_baseline.R` | Tests whether distortions depend on how the writer's own paragraph was rated. |
| `phase_2_distortion_by_input_condition.py` | Collates distortion estimates by input condition. |
| `phase_2_distortion_by_model.py` | Collates distortion estimates by model. |
| `phase_2_distortion_by_model_pairwise.R` | Tests pairwise differences in distortions between models. |
| `phase_2_distortion_by_proposition_leaning.R` | Estimates distortions separately for left- and right-leaning propositions. |
| `phase_2_distortion_nominal_variables.R` | Estimates distortions for nominal variables. |
| `phase_2_distortion_ordinal_variables.R` | Estimates distortions for ordinal variables. |
| `phase_2_distortion_plots.py` | Plots distortion estimates. |
| `phase_2_distortion_scale_variables.R` | Estimates distortions for scale variables. |
| `phase_2_random_effects_sensitivity.R` | Compares distortion estimates across random-effects structures. |
| `phase_2_random_effects_sensitivity_plots.R` | Summarizes and plots the random-effects comparison. |

### Phase 2 Rating-Protocol Robustness

Scripts testing whether main-study distortion estimates are robust to the rating protocol (e.g. fatigue, rating order, speeding).

| File | Description |
|------|-------------|
| `phase_2_rating_robustness.R` | Re-estimates distortions under alternative rating-protocol specifications. |
| `phase_2_rating_robustness_plots.py` | Plots the rating-protocol robustness results. |

### Phase 2 Writer-Engagement Robustness

Scripts testing whether main-study distortion estimates are robust to excluding writing on issues that writers reported knowing or caring little about.

| File | Description |
|------|-------------|
| `phase_2_writer_engagement_robustness.R` | Re-estimates distortions excluding writing on issues writers knew or cared little about. |
| `phase_2_writer_engagement_robustness_plots.py` | Plots the writer-engagement robustness results. |

### Phase 2 Perception Accuracy

Scripts comparing reader perceptions of writers with the writers' own Phase 1 self-reports.

| File | Description |
|------|-------------|
| `phase_2_perception_accuracy.py` | Measures how well reader perceptions match writer self-reports. |
| `phase_2_perception_accuracy_plots.py` | Plots the perception accuracy results. |

## Follow-Up Studies

### `2_disclaimer_study`

Scripts for the disclaimer-condition follow-up analyses.

| File | Description |
|------|-------------|
| `phase_1_distortion_tolerance.py` | Summarizes writer tolerance for distortions by disclaimer condition. |
| `phase_1_paragraph_preference.R` | Analyzes writer preferences by disclaimer condition. |
| `phase_1_preference_reasons_coding.py` | Codes free-text preference reasons with an LLM (see [Preference Reasons](#preference-reasons)). |
| `phase_1_preference_reasons_summary.py` | Summarizes the coded preference reasons. |

### `3_mitigation_study`

Scripts for the mitigation-strategy follow-up analyses.

| File | Description |
|------|-------------|
| `phase_1_paragraph_preference.R` | Analyzes writer preferences by mitigation condition. |
| `phase_1_preference_reasons_coding.py` | Codes free-text preference reasons with an LLM (see [Preference Reasons](#preference-reasons)). |
| `phase_1_preference_reasons_summary.py` | Summarizes the coded preference reasons. |
| `phase_2_distortion_nominal_variables.R` | Estimates distortions for nominal variables by mitigation condition. |
| `phase_2_distortion_ordinal_variables.R` | Estimates distortions for ordinal variables by mitigation condition. |
| `phase_2_distortion_plots.py` | Plots distortion estimates by mitigation condition. |
| `phase_2_distortion_scale_variables.R` | Estimates distortions for scale variables by mitigation condition. |
| `phase_2_distortion_side_effects.R` | Summarizes side effects of the mitigation strategies. |
| `phase_2_distribution_plots.py` | Plots writer versus model annotation distributions by mitigation condition. |
| `phase_2_distribution_variables.R` | Tests for differences between writer and model annotation distributions. |
| `phase_2_rater_agreement.R` | Computes inter-rater agreement for all annotation attributes. |
| `phase_2_mitigation_side_effects_plots.py` | Plots distortion reductions against side effects of the mitigation strategies. |

### `4_trust_study`

Scripts for the trust follow-up analyses, which measure reader trust in AI-assisted versus human writing.

| File | Description |
|------|-------------|
| `trust_regressions_full_analyses.R` | Estimates the effect of AI assistance on reader trust. |
| `plots.py` | Plots the trust study results. |

### `5_persuasion_study`

Scripts for the persuasion follow-up analyses, which measure stance shift after reading AI-assisted versus human writing.

| File | Description |
|------|-------------|
| `persuasion_regressions_full_analyses.R` | Estimates the effect of AI assistance on persuasiveness. |
| `plots.py` | Plots the persuasion study results. |

## Shared Utilities

### `utils_py`

Shared Python helpers used by multiple plotting and summary scripts.

| File | Description |
|------|-------------|
| `demo_paths.py` | Resolves output directories for full and demo runs. |
| `plotting_utils.py` | Plotting helpers. |
| `preference_reasons.py` | Codes free-text preference reasons with an LLM. |
| `preference_reasons_summary.py` | Summarizes and plots coded preference reasons. |
| `text_metrics.py` | Computes paragraph text metrics and distinctive words. |
| `variable_definitions.py` | Defines shared variable lists and factor levels. |

### `utils_r`

Shared R helpers used by multiple analysis scripts.

| File | Description |
|------|-------------|
| `data_loading.R` | Loads Phase 2 annotations and builds the standard data splits. |
| `demo_paths.R` | Resolves output directories for full and demo runs. |
| `rater_agreement.R` | Computes inter-rater agreement. |
| `variable_definitions.R` | Defines shared variable lists and factor levels. |

### `utils_files`

Shared non-code inputs.

| File | Description |
|------|-------------|
| `preference_reason_codebook.json` | Codebook for coding free-text preference reasons. |