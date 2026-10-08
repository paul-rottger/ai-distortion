#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: BY HUMAN BASELINE
#
# Secondary analysis: does the size of the AI distortion depend on how the
# writer's own paragraph was rated (the human baseline)?
#
# - Builds a split-half baseline per (writer, proposition) pair: writer-paragraph
#   ratings are randomly split into two halves; half A defines the baseline and
#   is dropped, half B (plus all model paragraph ratings) is the outcome data.
#   This keeps baseline measurement error independent of the outcome and avoids
#   spurious regression-to-the-mean interactions.
# - Fits beta regressions (as in phase_2_distortion_scale_variables.R) with a
#   paragraph_type_ x baseline_z interaction and rater random intercepts.
# - Interaction is on the logit scale (beyond mechanical ceiling/floor effects);
#   AMEs at baseline quantiles are on the 0-100 response scale (include them).
# - Fits cumulative link models for ordinal variables with the same interaction.
#   Unlike phase_2_distortion_ordinal_variables.R, these use rater-clustered
#   robust SEs instead of rater random intercepts, since clmm with the
#   interaction is computationally infeasible on the full data. The ordinal
#   baseline is the mean level code (1 = lowest level) in half A. Reports ORs
#   of model vs writer at baseline quantiles.
# - Pass "scale_only" or "ordinal_only" to run one part of the analysis.
# - Writes results to results/main_phase_2_distortion/<split>/by_baseline/.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
  library(ordinal)
  library(sandwich)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility
set.seed(123)

# Parse command-line flags
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
run_scale <- !("ordinal_only" %in% args)
run_ordinal <- !("scale_only" %in% args)
RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_distortion")

# Baseline quantiles at which to report AMEs
BASELINE_QUANTILES <- c(q10 = 0.1, q50 = 0.5, q90 = 0.9)

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# Assign writer-paragraph ratings to random halves once, so that the split is
# identical across attributes and data splits
list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      group_by(writer_id, proposition_id, paragraph_type) %>%
      mutate(
        baseline_half = if (first(paragraph_type) == "writer") {
          sample(rep(c("A", "B"), length.out = n()))
        } else {
          NA_character_
        }
      ) %>%
      ungroup() %>%
      mutate(across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE)))
  }
), envir = environment())

# Build regression data for one attribute: baseline from half A writer ratings,
# outcome from half B writer ratings plus all model paragraph ratings.
# Ordered factors are scored by their level code (1 = lowest level).
make_baseline_split <- function(df, attribute) {
  baselines <- df %>%
    filter(baseline_half == "A", !is.na(.data[[attribute]])) %>%
    group_by(writer_id, proposition_id) %>%
    summarise(baseline = mean(as.numeric(.data[[attribute]])), .groups = "drop")

  baseline_mean <- mean(baselines$baseline)
  baseline_sd <- sd(baselines$baseline)

  baselines <- baselines %>%
    mutate(baseline_z = (baseline - baseline_mean) / baseline_sd)

  model_df <- df %>%
    filter(is.na(baseline_half) | baseline_half == "B", !is.na(.data[[attribute]])) %>%
    inner_join(baselines, by = c("writer_id", "proposition_id"))

  list(
    model_df = model_df,
    baselines = baselines,
    baseline_mean = baseline_mean,
    baseline_sd = baseline_sd
  )
}

# =============================================================================
# ANALYSIS: BETA REGRESSIONS WITH BASELINE MODERATION
# =============================================================================

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

fit_beta_moderation <- function(df, outcome, baselines, baseline_mean, baseline_sd,
                                predictor = "paragraph_type_", moderator = "baseline_z",
                                random = "(1 | rater_id)") {
  # 0–100 -> proportion, then squeeze to (0,1)
  y <- df[[outcome]] / 100
  y <- pmin(pmax(y, 0), 1)
  df$y <- squeeze01(y)

  # Build model
  form <- as.formula(paste0("y ~ ", predictor, " * ", moderator, " + ", random))
  model <- glmmTMB(form, data = df, family = beta_family(link = "logit"))

  # Fixed effects (all terms, incl. interaction)
  tidy_fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p = p.value
    ) %>%
    select(term, odds_ratio, or_low, or_high, statistic, p)

  # AMEs of model vs writer at baseline quantiles (pair-level distribution)
  q_values <- quantile(baselines[[moderator]], probs = BASELINE_QUANTILES, names = FALSE)

  ame <- avg_comparisons(
    model,
    variables = setNames(list("reference"), predictor),
    newdata = datagrid(model = model, baseline_z = q_values),
    by = moderator,
    type = "response",
    re.form = NA
  ) %>%
    as_tibble() %>%
    mutate(
      term = paste0(predictor, sub(" .*", "", contrast)),
      baseline_quantile = names(BASELINE_QUANTILES)[match(round(.data[[moderator]], 8), round(q_values, 8))],
      baseline_z = .data[[moderator]],
      baseline_raw = baseline_mean + baseline_z * baseline_sd,
      ame = estimate * 100,
      ame_low = conf.low * 100,
      ame_high = conf.high * 100
    ) %>%
    select(term, baseline_quantile, baseline_z, baseline_raw, ame, ame_low, ame_high)

  list(model = model, tidy_fixed = tidy_fixed, ame = ame)
}

# =============================================================================
# ANALYSIS: ORDINAL REGRESSIONS WITH BASELINE MODERATION
# =============================================================================

fit_ordinal_moderation <- function(df, outcome, baselines, baseline_mean, baseline_sd,
                                   predictor = "paragraph_type_", moderator = "baseline_z",
                                   cluster = "rater_id") {
  # Cumulative link model (logit link) with rater-clustered robust SEs
  form <- as.formula(paste0(outcome, " ~ ", predictor, " * ", moderator))
  model <- clm(
    form,
    data = df,
    link = "logit",
    control = clm.control(maxIter = 200, gradTol = 1e-4)
  )
  V <- vcovCL(model, cluster = df[[cluster]])

  # Keep only predictor and moderator terms, not threshold/cutpoint parameters
  terms <- names(model$beta)
  estimate <- model$beta
  std_error <- sqrt(diag(V)[terms])

  tidy_fixed <- tibble(term = terms, estimate = estimate, std_error = std_error) %>%
    transmute(
      term = term,
      odds_ratio = exp(estimate),
      or_low = exp(estimate - 1.96 * std_error),
      or_high = exp(estimate + 1.96 * std_error),
      statistic = estimate / std_error,
      p = 2 * pnorm(-abs(statistic))
    )

  # ORs of model vs writer at baseline quantiles: exp(b_model + b_interaction * z)
  main_term <- paste0(predictor, "model")
  int_term <- paste0(main_term, ":", moderator)
  b <- model$beta[c(main_term, int_term)]
  V_sub <- V[c(main_term, int_term), c(main_term, int_term)]
  q_values <- quantile(baselines[[moderator]], probs = BASELINE_QUANTILES, names = FALSE)

  or_by_baseline <- tibble(
    term = main_term,
    baseline_quantile = names(BASELINE_QUANTILES),
    baseline_z = q_values,
    baseline_raw = baseline_mean + q_values * baseline_sd
  ) %>%
    mutate(
      estimate = b[1] + b[2] * baseline_z,
      std_error = sqrt(V_sub[1, 1] + baseline_z^2 * V_sub[2, 2] + 2 * baseline_z * V_sub[1, 2]),
      odds_ratio = exp(estimate),
      or_low = exp(estimate - 1.96 * std_error),
      or_high = exp(estimate + 1.96 * std_error),
      p = 2 * pnorm(-abs(estimate / std_error))
    ) %>%
    select(term, baseline_quantile, baseline_z, baseline_raw, odds_ratio, or_low, or_high, p)

  list(model = model, tidy_fixed = tidy_fixed, or_by_baseline = or_by_baseline)
}

# =============================================================================
# ANALYSIS: RUN REGRESSIONS
# =============================================================================

run_regressions <- function(attribute) {
  print(paste("running baseline moderation regressions for:", attribute))

  summary_rows <- list()

  for (data_split in if (demo_mode) c("preferred") else c("preferred", "edited", "unedited")) {
    split_data <- get_split_data(data_split)
    baseline_split <- make_baseline_split(split_data, attribute)
    model_df <- baseline_split$model_df

    if (demo_mode) {
      model_df <- model_df %>%
        slice_sample(n = min(1000, nrow(model_df)))
    }

    output_dir <- file.path(RESULTS_DIR, data_split, "by_baseline")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    results <- fit_beta_moderation(
      model_df,
      outcome = attribute,
      baselines = baseline_split$baselines,
      baseline_mean = baseline_split$baseline_mean,
      baseline_sd = baseline_split$baseline_sd
    )

    write_csv(results$tidy_fixed, file.path(output_dir, paste0(attribute, "_coefficients.csv")))
    write_csv(results$ame, file.path(output_dir, paste0(attribute, "_ame_by_baseline.csv")))

    interaction <- results$tidy_fixed %>%
      filter(str_detect(term, ":")) %>%
      select(interaction_or = odds_ratio, interaction_or_low = or_low,
             interaction_or_high = or_high, interaction_p = p)

    ame_wide <- results$ame %>%
      select(baseline_quantile, baseline_raw, ame, ame_low, ame_high) %>%
      pivot_wider(
        names_from = baseline_quantile,
        values_from = c(baseline_raw, ame, ame_low, ame_high),
        names_glue = "{.value}_{baseline_quantile}"
      )

    summary_rows[[data_split]] <- tibble(
      attribute = attribute,
      data_split = data_split,
      n_ratings = nrow(model_df),
      n_pairs = n_distinct(paste(model_df$writer_id, model_df$proposition_id))
    ) %>%
      bind_cols(interaction, ame_wide)
  }

  bind_rows(summary_rows)
}

run_ordinal_regressions <- function(attribute) {
  print(paste("running ordinal regressions with baseline moderation for:", attribute))

  summary_rows <- list()

  for (data_split in if (demo_mode) c("preferred") else c("preferred", "edited", "unedited")) {
    split_data <- get_split_data(data_split)

    # Drop "Other" only when fitting writer_education model
    if (attribute == "writer_education") {
      split_data <- split_data %>%
        filter(writer_education != "Other")
    }

    baseline_split <- make_baseline_split(split_data, attribute)
    model_df <- baseline_split$model_df

    if (demo_mode) {
      model_df <- model_df %>%
        slice_sample(n = min(1000, nrow(model_df)))
    }

    # Drop unused outcome levels (e.g. education "Other", rare levels in demo samples)
    model_df[[attribute]] <- droplevels(model_df[[attribute]])

    output_dir <- file.path(RESULTS_DIR, data_split, "by_baseline")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    results <- fit_ordinal_moderation(
      model_df,
      outcome = attribute,
      baselines = baseline_split$baselines,
      baseline_mean = baseline_split$baseline_mean,
      baseline_sd = baseline_split$baseline_sd
    )

    write_csv(results$tidy_fixed, file.path(output_dir, paste0(attribute, "_coefficients.csv")))
    write_csv(results$or_by_baseline, file.path(output_dir, paste0(attribute, "_or_by_baseline.csv")))

    interaction <- results$tidy_fixed %>%
      filter(str_detect(term, ":")) %>%
      select(interaction_or = odds_ratio, interaction_or_low = or_low,
             interaction_or_high = or_high, interaction_p = p)

    or_wide <- results$or_by_baseline %>%
      select(baseline_quantile, baseline_raw, odds_ratio, or_low, or_high) %>%
      pivot_wider(
        names_from = baseline_quantile,
        values_from = c(baseline_raw, odds_ratio, or_low, or_high),
        names_glue = "{.value}_{baseline_quantile}"
      )

    summary_rows[[data_split]] <- tibble(
      attribute = attribute,
      data_split = data_split,
      n_ratings = nrow(model_df),
      n_pairs = n_distinct(paste(model_df$writer_id, model_df$proposition_id))
    ) %>%
      bind_cols(interaction, or_wide)
  }

  bind_rows(summary_rows)
}

if (demo_mode) {
  message("Running in demo mode on n=1000 samples from the preferred split.")
}

# Loop through all scale attributes
if (run_scale) {
  scale_summary <- map_dfr(rating_attributes, run_regressions)
  write_csv(scale_summary, file.path(RESULTS_DIR, "by_baseline_summary.csv"))
}

# Loop through all ordinal attributes
if (run_ordinal) {
  ordinal_summary <- map_dfr(ordinal_vars, run_ordinal_regressions)
  write_csv(ordinal_summary, file.path(RESULTS_DIR, "by_baseline_ordinal_summary.csv"))
}
