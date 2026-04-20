#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: ORDINAL VARIABLES
# 
# Estimates ordinal distortion effects in writer and model paragraphs.
#
# - Fits cumulative link mixed models for ordinal outcomes.
# - Computes model-level and input-condition contrasts against writer baselines.
# - Runs analyses on unedited and edited subsets.
# - Writes ordinal distortion result tables to results/main_phase_2_distortion/.
# 
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
  library(ordinal)
  library(parallel)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility
# ===== RANDOM SEED ----
set.seed(123)

# Parse command-line flags
# ===== COMMAND-LINE FLAGS ----
args <- commandArgs(trailingOnly = TRUE)
debug_mode <- "debug" %in% args

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      mutate(
        model_           = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
        input_condition_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)), ref = "writer"),
        across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
      )
  }
), envir = environment())

# =============================================================================
# ANALYSIS: ORDINAL REGRESSION MODELS
# =============================================================================

# ===== ORDINAL LOGISTIC REGRESSION (BY TYPE) ----

fit_ordinal_logit <- function(df, outcome, predictor = "paragraph_type_", random = "(1 | rater_id)") {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
    mutate(
      rater_id = as.factor(rater_id)
    )

  if (predictor == "paragraph_type_") {
    model_df <- model_df %>%
      mutate(paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer"))
  }

  if (predictor == "model_") {
    model_df <- model_df %>%
      mutate(model_ = relevel(as.factor(model_), ref = "writer"))
  }

  if (predictor == "input_condition_") {
    model_df <- model_df %>%
      mutate(input_condition_ = relevel(as.factor(input_condition_), ref = "writer"))
  }

  # Drop "Other" only when fitting writer_education model
  if (outcome == "writer_education") {
    model_df <- model_df %>%
      filter(writer_education != "Other")
  }

  # Cumulative link mixed model (logit link)
  form <- as.formula(paste0(outcome, " ~ ", predictor, " + ", random))
  model <- clmm(
    form,
    data = model_df,
    link = "logit",
    Hess = TRUE,
    nAGQ = 0,
    control = clmm.control(maxIter = 200, gradTol = 1e-4)
  )

  coef_tbl <- as_tibble(summary(model)$coefficients, rownames = "term")

  # Keep only predictor term(s), not threshold/cutpoint parameters
  tidy_fixed <- coef_tbl %>%
    filter(str_starts(term, predictor)) %>%
    transmute(
      term = term,
      estimate = Estimate,
      std_error = `Std. Error`,
      statistic = `z value`,
      p = `Pr(>|z|)`,
      conf.low = estimate - 1.96 * std_error,
      conf.high = estimate + 1.96 * std_error,
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p_value = p
    ) %>%
    dplyr::select(term, odds_ratio, or_low, or_high, statistic, p, p_value)

  list(model = model, tidy_fixed = tidy_fixed)
}

run_ordinal_regressions <- function(attribute) {
  print(paste("running ordinal logistic regression for:", attribute))

  for (data_split in c("unedited", "edited")) {
    for (predictor in list(
      #c("paragraph_type_", "by_type")
      c("model_", "by_model"),
      c("input_condition_", "by_input_condition")
    )) {
      split_data <- switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      )

      if (debug_mode) {
        split_data <- split_data %>%
          slice_sample(n = min(1000, nrow(split_data)))
      }

      dir.create(
        paste0("./results/main_phase_2_distortion/", data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- fit_ordinal_logit(
        split_data,
        outcome = attribute,
        predictor = predictor[1],
        random = "(1 | rater_id)"
      )

      write_csv(
        results$tidy_fixed,
        paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_", predictor[2], ".csv")
      )
    }
  }
}

if (debug_mode) {
  message("Running in debug mode on n=1000 samples from each data split.")
}

# Loop through all ordinal attributes and predictors
for (attribute in ordinal_vars) {
  run_ordinal_regressions(attribute)
}
