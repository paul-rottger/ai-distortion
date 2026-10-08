#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 ROBUSTNESS: WRITER ENGAGEMENT
#
# Tests whether writer-vs-model distortion estimates for scale attributes are
# robust to excluding writing on issues that writers reported knowing or caring
# little about.
#
# - Joins writers' Phase 1 self-reported issue knowledge, importance, and
#   confidence onto Phase 2 annotations by writer and proposition.
# - Excludes writer-proposition pairs (both writer and model paragraphs) that
#   score below the scale midpoint on each engagement measure in turn.
# - Re-estimates writer-vs-model AMEs with crossed reader, writer, and
#   proposition random intercepts, as in the main analysis.
# - Writes result tables to results/main_phase_2_robustness/.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(marginaleffects)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility
set.seed(123)

# Parse command-line flags
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_robustness")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

PHASE1_PATH <- "./data/main_phase_1/proposition_responses.csv"

# Exclusion threshold: writer-proposition pairs below the scale midpoint
ENGAGEMENT_THRESHOLD <- 50

RANDOM_EFFECTS <- "(1 | rater_id) + (1 | writer_id) + (1 | proposition_id)"

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# Writer self-reports (annotations.csv columns of the same name are reader perceptions)
writer_engagement <- read_csv(PHASE1_PATH, show_col_types = FALSE) %>%
  transmute(
    writer_id      = as.factor(writer_id),
    proposition_id = as.factor(proposition_id),
    self_knowledge  = writer_knowledge,
    self_importance = writer_importance,
    self_confidence = writer_confidence
  )

list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  PHASE1_PATH
), envir = environment())

data_robustness <- data_preferred %>%
  left_join(writer_engagement, by = c("writer_id", "proposition_id")) %>%
  mutate(
    low_knowledge  = self_knowledge < ENGAGEMENT_THRESHOLD,
    low_importance = self_importance < ENGAGEMENT_THRESHOLD,
    low_confidence = self_confidence < ENGAGEMENT_THRESHOLD
  )

stopifnot(!anyNA(data_robustness$self_knowledge))

# =============================================================================
# DESCRIPTIVES: EXCLUSIONS
# =============================================================================

engagement_flags <- c("low_knowledge", "low_importance", "low_confidence")

pairs_robustness <- data_robustness %>%
  distinct(writer_id, proposition_id, across(all_of(engagement_flags)))

exclusion_counts <- tibble(check = engagement_flags) %>%
  mutate(
    n_pairs_excluded = map_int(check, ~ sum(pairs_robustness[[.x]])),
    n_pairs_total = nrow(pairs_robustness),
    share_pairs_excluded = n_pairs_excluded / n_pairs_total,
    n_rows_excluded = map_int(check, ~ sum(data_robustness[[.x]])),
    n_rows_total = nrow(data_robustness),
    share_rows_excluded = n_rows_excluded / n_rows_total,
    threshold = paste0("self-report < ", ENGAGEMENT_THRESHOLD)
  )
print(exclusion_counts)
write_csv(exclusion_counts, file.path(RESULTS_DIR, "engagement_exclusion_counts.csv"))

# =============================================================================
# ANALYSIS: BETA REGRESSIONS UNDER ALTERNATIVE SPECIFICATIONS
# =============================================================================

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

# Fits beta regression of outcome on paragraph type and returns the
# writer-vs-model average marginal effect on the 0-100 scale
fit_beta_ame <- function(df, outcome) {
  df$y <- squeeze01(pmin(pmax(df[[outcome]] / 100, 0), 1))

  model <- glmmTMB(
    as.formula(paste("y ~ paragraph_type_ +", RANDOM_EFFECTS)),
    data = df,
    family = beta_family(link = "logit")
  )

  avg_comparisons(
    model,
    variables = list(paragraph_type_ = "reference"),
    type = "response",
    re.form = NA
  ) %>%
    as_tibble() %>%
    transmute(
      ame = estimate * 100,
      ame_low = conf.low * 100,
      ame_high = conf.high * 100,
      p = p.value,
      n = nrow(df)
    )
}

specifications <- list(
  full              = function(d) d,
  no_low_knowledge  = function(d) filter(d, !low_knowledge),
  no_low_importance = function(d) filter(d, !low_importance),
  no_low_confidence = function(d) filter(d, !low_confidence)
)

ame_results <- list()

for (attribute in rating_attributes) {
  print(paste("running engagement robustness regressions for:", attribute))

  attribute_data <- data_robustness
  if (demo_mode) {
    attribute_data <- attribute_data %>% slice_sample(n = min(1000, nrow(attribute_data)))
  }

  for (spec_name in names(specifications)) {
    ame_results[[length(ame_results) + 1]] <- fit_beta_ame(
      specifications[[spec_name]](attribute_data), attribute
    ) %>%
      mutate(attribute = attribute, specification = spec_name, .before = 1)
  }
}

write_csv(bind_rows(ame_results), file.path(RESULTS_DIR, "engagement_ame_by_specification.csv"))

if (demo_mode) {
  message("Ran in demo mode on n=1000 samples per attribute.")
}
