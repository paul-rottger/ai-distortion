#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: SCALE VARIABLES
# 
# Estimates scale-based distortion effects for writer and model paragraphs.
#
# - Fits beta regressions for scale outcomes, with crossed reader, writer and
#   proposition random intercepts (as pre-registered).
# - Computes average marginal effects for paragraph-type, model and
#   input-condition comparisons.
# - Runs analyses on unedited, edited, and preferred subsets.
# - Writes scale distortion result tables to results/main_phase_2_distortion/.
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
  library(parallel)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility (parallel-safe RNG streams)
RNGkind("L'Ecuyer-CMRG")
set.seed(123)

# Parse command-line flags
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_distortion")

# Crossed random intercepts for readers, writers and propositions
RANDOM_EFFECTS <- "(1 | rater_id) + (1 | writer_id) + (1 | proposition_id)"
N_CORES <- min(10, detectCores())

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      mutate(
        model_           = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
        input_condition_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)), ref = "writer")
      )
  }
), envir = environment())

# =============================================================================
# ANALYSIS: BETA REGRESSIONS FOR SCALE VARIABLES
# =============================================================================

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

fit_beta <- function(df, outcome, predictor, random) {
  # 0–100 -> proportion, then squeeze to (0,1)
  y <- df[[outcome]] / 100
  y <- pmin(pmax(y, 0), 1)
  y <- squeeze01(y)

  # Build model
  form <- as.formula(paste0("y ~ ", predictor, " + ", random))
  model <- glmmTMB(form, data = df, family = beta_family(link = "logit"))

  # Fixed effects
  tidy_fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p = p.value
    ) %>%
    select(term, odds_ratio, or_low, or_high, statistic, p)

  # Compute average marginal effects. If a random-effect variance collapses to
  # zero (only seen on small demo subsamples), marginaleffects cannot build the
  # covariance matrix; fall back to point estimates without CIs.
  ame_args <- list(
    model,
    variables = setNames(list("reference"), predictor),
    type = "response",
    re.form = NA
  )
  ame <- tryCatch(
    as_tibble(do.call(avg_comparisons, ame_args)),
    error = function(e) {
      warning(sprintf("AME CIs unavailable for %s ~ %s: %s", outcome, predictor, conditionMessage(e)), call. = FALSE)
      as_tibble(do.call(avg_comparisons, c(ame_args, vcov = FALSE))) %>%
        mutate(conf.low = NA_real_, conf.high = NA_real_)
    }
  ) %>%
    mutate(
      term = paste0(predictor, sub(" .*", "", contrast)),
      ame = estimate * 100,
      ame_low = conf.low * 100,
      ame_high = conf.high * 100,
    ) %>%
    select(term, ame, ame_low, ame_high)

  tidy_fixed <- tidy_fixed %>% left_join(ame, by = "term")

  list(model = model, tidy_fixed = tidy_fixed)
}

# =============================================================================
# ANALYSIS: RUN SCALE VARIABLE REGRESSIONS
# =============================================================================

run_regressions <- function(attribute) {
  print(paste("running regressions for:", attribute))

  for (data_split in if (demo_mode) c("preferred") else c("preferred", "edited", "unedited")) {
    for (predictor in list(
      c("paragraph_type_", "by_type"),
      c("model_", "by_model"),
      c("input_condition_", "by_input")
    )) {
      split_data <- switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      )

      if (demo_mode) {
        split_data <- split_data %>%
          slice_sample(n = min(1000, nrow(split_data)))
      }

      dir.create(
        file.path(RESULTS_DIR, data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- fit_beta(split_data,
        outcome = attribute,
        predictor = predictor[1],
        random = RANDOM_EFFECTS
      )
      write_csv(
        results$tidy_fixed,
        file.path(RESULTS_DIR, data_split, paste0(attribute, "_", predictor[2], ".csv"))
      )
    }
  }
}

# Loop through all rating attributes
if (demo_mode) {
  message("Running in demo mode on n=1000 samples from each data split.")
}

# Fit attributes in parallel; stop if any attribute failed
status <- mclapply(rating_attributes, run_regressions, mc.cores = N_CORES, mc.set.seed = TRUE)
failed <- vapply(status, inherits, logical(1), what = "try-error")
if (any(failed)) {
  stop("Regressions failed for: ", paste(rating_attributes[failed], collapse = ", "))
}
