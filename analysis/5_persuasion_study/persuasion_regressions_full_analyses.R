#!/usr/bin/env Rscript

# =============================================================================
# FOLLOWUP PERSUASION STUDY - FULL MIXED-EFFECTS ANALYSES
#
# Implements (per preregistration):
# - RQ1 primary: stance_shift ~ paragraph_type + (1 | reader_id) + (1 | proposition_id)
#     paragraph_type in {control, human, ai}, reference = control.
#     Primary contrast: AI vs human. Secondary: human vs control, ai vs control.
# - RQ2 binned: stance_shift ~ condition + (1 | reader_id) + (1 | proposition_id)
#     condition in {control, human-low, human-high, ai-low, ai-high}, ref = control.
#     Dose-response: (ai-high - human-high) - (ai-low - human-low).
# - RQ2 continuous (secondary, treated only):
#     stance_shift ~ paragraph_type * distortion_magnitude + (crossed)
# - Secondary: stance alignment, presentation order, clear-stance subset,
#     movement-towards-stance outcome (stance_shift_dist).
# - Exploratory: separate models adding each of the 20 paragraph attributes.
#
# Outcome is a signed stance shift on the 0-100 policy-attitude scale, oriented
# towards the position expressed in the paragraph. Modelled with a linear mixed
# model via glmmTMB(family = gaussian()); AMEs are in attitude points.
#
# Crossed reader + proposition random intercepts, with fallback to a reader-only
# random intercept should the crossed model fail to converge.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/demo_paths.R")

set.seed(123)

demo_mode <- parse_demo_mode()

INPUT_ANNOTATIONS_PATH <- "./data/followup_persuasion/annotations.csv"
INPUT_PAIRS_PATH <- "./data/followup_persuasion/paragraph_pairs.csv"
RESULTS_DIR <- get_results_dir(demo_mode, "followup_persuasion")
FIGURES_DIR <- get_figures_dir(demo_mode, "followup_persuasion")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

# Crossed and reader-only random-effects specifications.
RE_CROSSED <- "(1 | reader_id) + (1 | proposition_id)"
RE_READER <- "(1 | reader_id)"

# =============================================================================
# HELPERS
# =============================================================================

require_columns <- function(df, cols, label) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(paste0(
      "Missing required columns in ", label, ": ",
      paste(missing, collapse = ", ")
    ))
  }
}

# A model is usable if its fixed-effects table has all-finite estimates and SEs.
fixed_table <- function(model) {
  if (is.null(model)) return(NULL)
  tryCatch(broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE),
           error = function(e) NULL)
}

usable_fit <- function(model) {
  td <- fixed_table(model)
  !is.null(td) && all(is.finite(td$estimate)) && all(is.finite(td$std.error))
}

# A model has converged cleanly if it is usable and the Hessian is positive definite.
converged_fit <- function(model) usable_fit(model) && isTRUE(model$sdr$pdHess)

# Fit a linear mixed model. Prefer the crossed-effects specification; if it fails
# to converge, fall back to a reader-only random intercept (as preregistered).
# Only if neither converges cleanly do we accept a usable-but-not-pdHess fit.
fit_lmm <- function(df, fixed_rhs, model_name, random = RE_CROSSED) {
  build <- function(re) as.formula(paste0("stance_shift ~ ", fixed_rhs, " + ", re))
  fit_one <- function(re) tryCatch(
    suppressWarnings(glmmTMB(build(re), data = df, family = gaussian())),
    error = function(e) NULL
  )

  m_cross <- fit_one(random)
  m_reader <- NULL

  if (converged_fit(m_cross)) {
    model <- m_cross; used_random <- random
  } else {
    if (random != RE_READER) {
      message("  [", model_name, "] crossed model did not converge cleanly; ",
              "falling back to reader-only random intercept.")
      m_reader <- fit_one(RE_READER)
    }
    if (converged_fit(m_reader)) {
      model <- m_reader; used_random <- RE_READER
    } else if (usable_fit(m_cross)) {
      model <- m_cross; used_random <- random
    } else if (usable_fit(m_reader)) {
      model <- m_reader; used_random <- RE_READER
    } else {
      stop("Model '", model_name, "' failed to fit.")
    }
  }

  fixed <- fixed_table(model) %>%
    mutate(model = model_name, random_effects = used_random, p = p.value) %>%
    select(model, random_effects, term, estimate, conf.low, conf.high, statistic, p)

  list(model = model, fixed = fixed, random_effects = used_random)
}

# AME for a level-vs-reference categorical contrast (estimate already in points).
compute_ame_contrast <- function(model, variable, model_name, by_var = NULL) {
  args <- list(
    model = model,
    variables = setNames(list("reference"), variable),
    type = "response",
    re.form = NA
  )
  if (!is.null(by_var)) args$by <- by_var

  out <- do.call(avg_comparisons, args) %>%
    as_tibble() %>%
    mutate(
      model = model_name,
      term = paste0(variable, "_", gsub(" - ", "_vs_", contrast)),
      ame = estimate,
      ame_low = conf.low,
      ame_high = conf.high,
      p = p.value
    )

  keep <- c("model", "term", "ame", "ame_low", "ame_high", "statistic", "p")
  if (!is.null(by_var) && by_var %in% names(out)) {
    keep <- c("model", by_var, "term", "ame", "ame_low", "ame_high", "statistic", "p")
  }
  out %>% select(all_of(keep))
}

# AME of a pairwise contrast between two specified levels (e.g. ai vs human).
compute_ame_pair <- function(model, variable, lo, hi, model_name, term_label, by_var = NULL) {
  args <- list(
    model = model,
    variables = setNames(list(c(lo, hi)), variable),
    type = "response",
    re.form = NA
  )
  if (!is.null(by_var)) args$by <- by_var

  out <- do.call(avg_comparisons, args) %>%
    as_tibble() %>%
    mutate(model = model_name, term = term_label,
           ame = estimate, ame_low = conf.low, ame_high = conf.high, p = p.value)

  keep <- c("model", "term", "ame", "ame_low", "ame_high", "statistic", "p")
  if (!is.null(by_var) && by_var %in% names(out)) {
    keep <- c("model", by_var, "term", "ame", "ame_low", "ame_high", "statistic", "p")
  }
  out %>% select(all_of(keep))
}

# AME (slope) for a continuous predictor.
compute_ame_slope <- function(model, variable, model_name) {
  avg_slopes(model, variables = variable, type = "response", re.form = NA) %>%
    as_tibble() %>%
    mutate(model = model_name, term = variable,
           ame = estimate, ame_low = conf.low, ame_high = conf.high, p = p.value) %>%
    select(model, term, ame, ame_low, ame_high, statistic, p)
}

# Planned linear contrast among condition marginal means via hypotheses().
compute_hypothesis <- function(model, by_var, hypothesis, model_name, term_label) {
  preds <- avg_predictions(model, by = by_var, type = "response", re.form = NA)
  hypotheses(preds, hypothesis = hypothesis) %>%
    as_tibble() %>%
    transmute(
      model = model_name,
      term = term_label,
      ame = estimate,
      ame_low = conf.low,
      ame_high = conf.high,
      statistic = statistic,
      p = p.value
    )
}

# =============================================================================
# DATA PREP
# =============================================================================

raw <- read_csv(INPUT_ANNOTATIONS_PATH, show_col_types = FALSE) %>%
  mutate(
    row_in_file = row_number(),
    reader_id = as.character(rater_id),
    writer_id = as.character(writer_id),
    proposition_id = as.character(proposition_id)
  ) %>%
  group_by(reader_id) %>%
  arrange(row_in_file, .by_group = TRUE) %>%
  mutate(presentation_order = row_number()) %>%
  ungroup()

require_columns(
  raw,
  c("condition_type", "source", "distortion_bin",
    "pre_support", "pre_bad_idea", "pre_good_consequences",
    "post_support", "post_bad_idea", "post_good_consequences",
    "policy_attitude_pre", "policy_attitude_post"),
  "annotations.csv"
)

# Recompute the 0-100 policy_attitude score from the three battery items, keyed
# towards policy support (reverse-score bad_idea); fall back to provided columns.
policy_attitude <- function(support, bad_idea, good_consequences, provided) {
  recomputed <- (support + (100 - bad_idea) + good_consequences) / 3
  coalesce(recomputed, provided)
}

# Demo mode: sub-sample readers (keep all five cycles per sampled reader).
if (demo_mode) {
  sampled_readers <- raw %>%
    distinct(reader_id) %>%
    slice_sample(n = min(1000, n_distinct(raw$reader_id))) %>%
    pull(reader_id)
  raw <- raw %>% filter(reader_id %in% sampled_readers)
  message("Running in demo mode on ", length(sampled_readers), " readers.")
}

# Load original-study paragraph pairs: perceived stance + 20 attributes + deltas.
delta_cols <- paste0("delta_", rating_attributes)
human_cols <- paste0(rating_attributes, "_human")
ai_cols <- paste0(rating_attributes, "_ai")

pairs <- read_csv(INPUT_PAIRS_PATH, show_col_types = FALSE) %>%
  mutate(writer_id = as.character(writer_id),
         proposition_id = as.character(proposition_id))

require_columns(
  pairs,
  c("writer_id", "proposition_id", "writer_stance_human", "writer_stance_ai",
    delta_cols, human_cols, ai_cols),
  "pairs_with_deltas.csv"
)

pairs <- pairs %>%
  mutate(distortion_magnitude = rowMeans(across(all_of(delta_cols), ~ abs(.x)), na.rm = TRUE))

# Long format: one row per (writer, proposition, human/ai) with perceived stance,
# distortion magnitude, and the 20 attribute ratings.
pairs_long <- pairs %>%
  select(writer_id, proposition_id, distortion_magnitude,
         writer_stance_human, writer_stance_ai,
         all_of(human_cols), all_of(ai_cols)) %>%
  pivot_longer(
    cols = -c(writer_id, proposition_id, distortion_magnitude),
    names_to = c(".value", "join_source"),
    names_pattern = "(.*)_(human|ai)"
  )

# Build the analysis frame.
analysis_data <- raw %>%
  mutate(
    is_control = condition_type == "static_control",
    # 3-level paragraph type (control / human / ai) and join key for treated rows.
    paragraph_type = if_else(is_control, "control", source),
    join_source = if_else(is_control, NA_character_, source),
    bin_short = recode(distortion_bin,
                       low_distortion = "low", high_distortion = "high",
                       .default = NA_character_),
    policy_attitude_pre = policy_attitude(pre_support, pre_bad_idea,
                                          pre_good_consequences, policy_attitude_pre),
    policy_attitude_post = policy_attitude(post_support, post_bad_idea,
                                           post_good_consequences, policy_attitude_post)
  ) %>%
  left_join(pairs_long, by = c("writer_id", "proposition_id", "join_source")) %>%
  mutate(
    condition = if_else(is_control, "control", paste0(source, "-", bin_short)),
    # Direction towards the position expressed in the paragraph.
    direction = case_when(
      is_control ~ 1,
      !is.na(writer_stance) & writer_stance < 50 ~ -1,
      !is.na(writer_stance) ~ 1,
      TRUE ~ NA_real_
    ),
    delta_attitude = policy_attitude_post - policy_attitude_pre,
    stance_shift = direction * delta_attitude,
    # Distance between reader's prior attitude and the paragraph's perceived stance.
    stance_distance = if_else(is_control, NA_real_, abs(policy_attitude_pre - writer_stance)),
    # Movement towards the paragraph's perceived stance (continuous target).
    stance_shift_dist = if_else(
      is_control, NA_real_,
      abs(policy_attitude_pre - writer_stance) - abs(policy_attitude_post - writer_stance)
    )
  )

# Factor codings.
PT3_LEVELS <- c("control", "human", "ai")
PT2_LEVELS <- c("human", "ai")
COND5_LEVELS <- c("control", "human-low", "human-high", "ai-low", "ai-high")
COND4_LEVELS <- c("human-low", "human-high", "ai-low", "ai-high")

analysis_data <- analysis_data %>%
  mutate(
    paragraph_type3 = factor(paragraph_type, levels = PT3_LEVELS),
    paragraph_type2 = factor(paragraph_type, levels = PT2_LEVELS),
    condition5 = factor(condition, levels = COND5_LEVELS),
    condition4 = factor(condition, levels = COND4_LEVELS)
  )

treated_data <- analysis_data %>% filter(!is_control)

# =============================================================================
# RQ1: AI vs HUMAN (and absolute persuasion vs control)
# =============================================================================

message("Fitting RQ1 ...")

rq1_df <- analysis_data %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type3), !is.na(reader_id), !is.na(proposition_id))

rq1 <- fit_lmm(rq1_df, "paragraph_type3", "rq1_primary")

rq1_ame <- bind_rows(
  # human-vs-control and ai-vs-control (absolute persuasion).
  compute_ame_contrast(rq1$model, "paragraph_type3", "rq1_primary"),
  # ai-vs-human (primary quantity of interest).
  compute_ame_pair(rq1$model, "paragraph_type3", "human", "ai",
                   "rq1_primary", "paragraph_type_ai_vs_human")
)

# =============================================================================
# RQ2: DISTORTION MAGNITUDE AS MODERATOR
# =============================================================================

message("Fitting RQ2 (binned) ...")

rq2_df <- analysis_data %>%
  filter(!is.na(stance_shift), !is.na(condition5), !is.na(reader_id), !is.na(proposition_id))

rq2_bin <- fit_lmm(rq2_df, "condition5", "rq2_binned")

# Per-condition marginal means (transparency + plotting).
rq2_condition_means <- avg_predictions(rq2_bin$model, by = "condition5",
                                       type = "response", re.form = NA) %>%
  as_tibble() %>%
  transmute(model = "rq2_binned", condition = as.character(condition5),
            mean_stance_shift = estimate, conf.low, conf.high, p = p.value)

# condition5 factor order -> b1=control, b2=human-low, b3=human-high, b4=ai-low, b5=ai-high.
rq2_dose <- compute_hypothesis(
  rq2_bin$model, "condition5",
  "(b5 - b3) - (b4 - b2) = 0",
  "rq2_binned", "dose_response_ai_vs_human_high_minus_low"
)

rq2_ame <- bind_rows(
  # Each treated cell vs control (absolute persuasion).
  compute_ame_contrast(rq2_bin$model, "condition5", "rq2_binned"),
  # AI-vs-human within each distortion bin (for the whisker plot).
  compute_hypothesis(rq2_bin$model, "condition5", "b4 - b2 = 0",
                     "rq2_binned", "paragraph_type_ai_vs_human") %>%
    mutate(distortion_bin = "low_distortion"),
  compute_hypothesis(rq2_bin$model, "condition5", "b5 - b3 = 0",
                     "rq2_binned", "paragraph_type_ai_vs_human") %>%
    mutate(distortion_bin = "high_distortion"),
  # Dose-response contrast.
  rq2_dose
)

# ---- RQ2 continuous (secondary, treated only) ----
message("Fitting RQ2 (continuous) ...")

rq2_cont_df <- treated_data %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type2),
         !is.na(distortion_magnitude), !is.na(reader_id), !is.na(proposition_id))

rq2_cont <- fit_lmm(rq2_cont_df, "paragraph_type2 * distortion_magnitude", "rq2_continuous")

rq2_distortion_mean <- mean(rq2_cont_df$distortion_magnitude, na.rm = TRUE)
rq2_distortion_sd <- sd(rq2_cont_df$distortion_magnitude, na.rm = TRUE)

rq2_cont_growth_grid_1 <- datagrid(
  model = rq2_cont$model,
  distortion_magnitude = c(rq2_distortion_mean, rq2_distortion_mean + 1)
)
rq2_cont_growth_contrast_1 <- comparisons(
  rq2_cont$model,
  variables = list(paragraph_type2 = c("human", "ai")),
  newdata = rq2_cont_growth_grid_1,
  type = "response",
  re.form = NA
)
rq2_cont_growth_per_1 <- hypotheses(rq2_cont_growth_contrast_1, hypothesis = "b2 - b1 = 0") %>%
  as_tibble() %>%
  transmute(
    model = "rq2_continuous",
    term = "ai_advantage_growth_per_1_distortion",
    ame = estimate,
    ame_low = conf.low,
    ame_high = conf.high,
    statistic,
    p = p.value
  )

rq2_cont_growth_grid_1sd <- datagrid(
  model = rq2_cont$model,
  distortion_magnitude = c(rq2_distortion_mean, rq2_distortion_mean + rq2_distortion_sd)
)
rq2_cont_growth_contrast_1sd <- comparisons(
  rq2_cont$model,
  variables = list(paragraph_type2 = c("human", "ai")),
  newdata = rq2_cont_growth_grid_1sd,
  type = "response",
  re.form = NA
)
rq2_cont_growth_per_1sd <- hypotheses(rq2_cont_growth_contrast_1sd, hypothesis = "b2 - b1 = 0") %>%
  as_tibble() %>%
  transmute(
    model = "rq2_continuous",
    term = "ai_advantage_growth_per_1sd_distortion",
    ame = estimate,
    ame_low = conf.low,
    ame_high = conf.high,
    statistic,
    p = p.value
  )

rq2_cont_ai_advantage_growth <- bind_rows(rq2_cont_growth_per_1, rq2_cont_growth_per_1sd)

rq2_cont_ame <- bind_rows(
  compute_ame_pair(rq2_cont$model, "paragraph_type2", "human", "ai",
                   "rq2_continuous", "paragraph_type_ai_vs_human"),
  compute_ame_slope(rq2_cont$model, "distortion_magnitude", "rq2_continuous"),
  rq2_cont_ai_advantage_growth
)

# =============================================================================
# SECONDARY ANALYSES
# =============================================================================

message("Fitting secondary models ...")

secondary_fixed <- list()
secondary_ame <- list()

# 1. Stance alignment (treated only; paragraph_type reduces to human/ai).
sa_df <- treated_data %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type2),
         !is.na(stance_distance), !is.na(reader_id), !is.na(proposition_id))
sa <- fit_lmm(sa_df, "paragraph_type2 + stance_distance", "secondary_stance_alignment")
secondary_fixed[["stance_alignment"]] <- sa$fixed
secondary_ame[["stance_alignment"]] <- bind_rows(
  compute_ame_pair(sa$model, "paragraph_type2", "human", "ai",
                   "secondary_stance_alignment", "paragraph_type_ai_vs_human"),
  compute_ame_slope(sa$model, "stance_distance", "secondary_stance_alignment")
)

# 2. Presentation order (all cycles; 3-level paragraph_type).
po_df <- analysis_data %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type3),
         !is.na(presentation_order), !is.na(reader_id), !is.na(proposition_id))
po <- fit_lmm(po_df, "paragraph_type3 + presentation_order", "secondary_presentation_order")
secondary_fixed[["presentation_order"]] <- po$fixed
secondary_ame[["presentation_order"]] <- bind_rows(
  compute_ame_contrast(po$model, "paragraph_type3", "secondary_presentation_order"),
  compute_ame_pair(po$model, "paragraph_type3", "human", "ai",
                   "secondary_presentation_order", "paragraph_type_ai_vs_human"),
  compute_ame_slope(po$model, "presentation_order", "secondary_presentation_order")
)

# 3. Clear-stance subset: |paragraph_stance - 50| >= 10 for treated; controls kept.
clear_keep <- analysis_data %>%
  filter(is_control | (!is.na(writer_stance) & abs(writer_stance - 50) >= 10))

cs_rq1_df <- clear_keep %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type3), !is.na(reader_id), !is.na(proposition_id))
cs_rq1 <- fit_lmm(cs_rq1_df, "paragraph_type3", "secondary_clear_stance_rq1")
secondary_fixed[["clear_stance_rq1"]] <- cs_rq1$fixed
secondary_ame[["clear_stance_rq1"]] <- bind_rows(
  compute_ame_contrast(cs_rq1$model, "paragraph_type3", "secondary_clear_stance_rq1"),
  compute_ame_pair(cs_rq1$model, "paragraph_type3", "human", "ai",
                   "secondary_clear_stance_rq1", "paragraph_type_ai_vs_human")
)

cs_rq2_df <- clear_keep %>%
  filter(!is.na(stance_shift), !is.na(condition5), !is.na(reader_id), !is.na(proposition_id))
cs_rq2 <- fit_lmm(cs_rq2_df, "condition5", "secondary_clear_stance_rq2")
secondary_fixed[["clear_stance_rq2"]] <- cs_rq2$fixed
secondary_ame[["clear_stance_rq2"]] <- bind_rows(
  compute_ame_contrast(cs_rq2$model, "condition5", "secondary_clear_stance_rq2"),
  compute_hypothesis(cs_rq2$model, "condition5", "(b5 - b3) - (b4 - b2) = 0",
                     "secondary_clear_stance_rq2", "dose_response_ai_vs_human_high_minus_low")
)

clear_stance_cell_counts <- clear_keep %>%
  filter(!is.na(stance_shift)) %>%
  count(condition, name = "n") %>%
  arrange(condition)

# 4. Movement-towards-stance outcome (treated only).
#    Re-orient the outcome to stance_shift_dist by fitting on that column.
msd_rq1_df <- treated_data %>%
  filter(!is.na(stance_shift_dist), !is.na(paragraph_type2),
         !is.na(reader_id), !is.na(proposition_id)) %>%
  mutate(stance_shift = stance_shift_dist)
msd_rq1 <- fit_lmm(msd_rq1_df, "paragraph_type2", "secondary_movement_toward_stance_rq1")
secondary_fixed[["movement_rq1"]] <- msd_rq1$fixed
secondary_ame[["movement_rq1"]] <- compute_ame_pair(
  msd_rq1$model, "paragraph_type2", "human", "ai",
  "secondary_movement_toward_stance_rq1", "paragraph_type_ai_vs_human"
)

# RQ2 dose-response on the movement outcome (treated cells only; no control).
msd_rq2_df <- treated_data %>%
  filter(!is.na(stance_shift_dist), !is.na(condition4),
         !is.na(reader_id), !is.na(proposition_id)) %>%
  mutate(stance_shift = stance_shift_dist)
msd_rq2 <- fit_lmm(msd_rq2_df, "condition4", "secondary_movement_toward_stance_rq2")
secondary_fixed[["movement_rq2"]] <- msd_rq2$fixed
# condition4 order -> b1=human-low, b2=human-high, b3=ai-low, b4=ai-high.
secondary_ame[["movement_rq2"]] <- compute_hypothesis(
  msd_rq2$model, "condition4", "(b4 - b2) - (b3 - b1) = 0",
  "secondary_movement_toward_stance_rq2", "dose_response_ai_vs_human_high_minus_low"
)

secondary_fixed_tbl <- bind_rows(secondary_fixed)
secondary_ame_tbl <- bind_rows(secondary_ame)

# =============================================================================
# EXPLORATORY: ATTRIBUTE-BY-ATTRIBUTE MODELS (treated only)
# =============================================================================

message("Fitting exploratory attribute models ...")

exploratory_fixed <- list()
exploratory_ame <- list()

for (attr in rating_attributes) {
  model_name <- paste0("exploratory_", attr)
  message("  attribute: ", attr)

  df_attr <- treated_data %>%
    filter(!is.na(stance_shift), !is.na(paragraph_type2),
           !is.na(.data[[attr]]), !is.na(reader_id), !is.na(proposition_id))

  fitted <- fit_lmm(df_attr, paste0("paragraph_type2 + ", attr), model_name)
  exploratory_fixed[[attr]] <- fitted$fixed

  exploratory_ame[[attr]] <- compute_ame_slope(fitted$model, attr, model_name) %>%
    mutate(attribute = attr) %>%
    select(attribute, everything())
}

exploratory_fixed_tbl <- bind_rows(exploratory_fixed)
exploratory_ame_tbl <- bind_rows(exploratory_ame)

# Bonferroni correction across the 20 exploratory attribute tests.
attr_term_mask <- exploratory_fixed_tbl$term %in% rating_attributes
exploratory_fixed_tbl$p_bonferroni <- NA_real_
exploratory_fixed_tbl$p_bonferroni[attr_term_mask] <- p.adjust(
  exploratory_fixed_tbl$p[attr_term_mask], method = "bonferroni"
)
exploratory_fixed_tbl$significant_bonferroni <- !is.na(exploratory_fixed_tbl$p_bonferroni) &
  exploratory_fixed_tbl$p_bonferroni < 0.05

exploratory_ame_tbl <- exploratory_ame_tbl %>%
  mutate(p_bonferroni = p.adjust(p, method = "bonferroni"),
         significant_bonferroni = p_bonferroni < 0.05)

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

message("Saving outputs ...")

write_csv(rq1$fixed, file.path(RESULTS_DIR, "rq1_fixed_effects.csv"))
write_csv(rq1_ame, file.path(RESULTS_DIR, "rq1_ame.csv"))

write_csv(bind_rows(rq2_bin$fixed, rq2_cont$fixed),
          file.path(RESULTS_DIR, "rq2_fixed_effects.csv"))
write_csv(bind_rows(rq2_ame, rq2_cont_ame), file.path(RESULTS_DIR, "rq2_ame.csv"))
write_csv(rq2_condition_means, file.path(RESULTS_DIR, "rq2_condition_means.csv"))
write_csv(rq2_cont_ai_advantage_growth,
          file.path(RESULTS_DIR, "rq2_continuous_ai_advantage_growth.csv"))

write_csv(secondary_fixed_tbl, file.path(RESULTS_DIR, "secondary_fixed_effects.csv"))
write_csv(secondary_ame_tbl, file.path(RESULTS_DIR, "secondary_ame.csv"))
write_csv(clear_stance_cell_counts, file.path(RESULTS_DIR, "clear_stance_cell_counts.csv"))

write_csv(exploratory_fixed_tbl, file.path(RESULTS_DIR, "exploratory_attributes_fixed_effects.csv"))
write_csv(exploratory_ame_tbl, file.path(RESULTS_DIR, "exploratory_attributes_ame.csv"))

coverage <- analysis_data %>%
  summarise(
    n_rows = n(),
    n_readers = n_distinct(reader_id),
    n_propositions = n_distinct(proposition_id),
    n_control = sum(is_control),
    n_treated = sum(!is_control),
    n_treated_unmatched = sum(!is_control & is.na(writer_stance)),
    n_missing_stance_shift = sum(is.na(stance_shift)),
    n_missing_stance_distance_treated = sum(!is_control & is.na(stance_distance)),
    n_missing_distortion_magnitude_treated = sum(!is_control & is.na(distortion_magnitude)),
    mean_control_stance_shift = mean(stance_shift[is_control], na.rm = TRUE)
  )
write_csv(coverage, file.path(RESULTS_DIR, "data_join_coverage.csv"))

message("Saved outputs to: ", RESULTS_DIR)
