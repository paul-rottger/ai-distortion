#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 ROBUSTNESS: RATING PROTOCOL
#
# Tests whether writer-vs-model distortion estimates for scale attributes are
# robust to features of the rating protocol (fatigue, rating order, carryover,
# speeding, and straightlining).
#
# - Checks that paragraph type is balanced across presentation positions.
# - Summarizes time spent per paragraph by position.
# - Re-estimates writer-vs-model AMEs under alternative specifications:
#   first paragraph only, early vs late positions, previous-paragraph control,
#   and speeder and straightliner exclusions.
# - Estimates writer-vs-model AMEs by position from a type x position interaction.
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

# Exclusion thresholds
SPEEDER_RATER_QUANTILE <- 0.10        # raters in bottom 10% of median seconds per paragraph
SPEEDER_PARAGRAPH_SECONDS <- 60       # any paragraph rated in under 60 seconds
STRAIGHTLINER_QUANTILE <- 0.05        # bottom 5% of within-paragraph SD across scale items

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

add_protocol_variables <- function(data) {
  data %>%
    mutate(row_id = row_number()) %>%
    # Previous paragraph type (writer vs AI) within each rater's sequence
    group_by(rater_id) %>%
    arrange(paragraph_position, .by_group = TRUE) %>%
    mutate(
      previous_type = lag(if_else(paragraph_type == "writer", "writer", "ai")),
      previous_type = factor(replace_na(previous_type, "none"), levels = c("none", "writer", "ai")),
      rater_median_seconds = median(paragraph_seconds)
    ) %>%
    ungroup() %>%
    arrange(row_id) %>%
    # Within-paragraph SD across all scale items (low = straightlining)
    mutate(item_sd = apply(across(all_of(rating_attributes)), 1, sd, na.rm = TRUE)) %>%
    select(-row_id)
}

list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = add_protocol_variables
), envir = environment())

# Exclusion flags, with thresholds set on the analysed sample
speeder_rater_threshold <- data_preferred %>%
  distinct(rater_id, rater_median_seconds) %>%
  pull(rater_median_seconds) %>%
  quantile(SPEEDER_RATER_QUANTILE)
straightliner_threshold <- quantile(data_preferred$item_sd, STRAIGHTLINER_QUANTILE, na.rm = TRUE)

data_robustness <- data_preferred %>%
  mutate(
    speeder = rater_median_seconds <= speeder_rater_threshold | paragraph_seconds < SPEEDER_PARAGRAPH_SECONDS,
    straightliner = item_sd <= straightliner_threshold
  )

# =============================================================================
# DESCRIPTIVES: POSITION BALANCE, TIMING, EXCLUSIONS
# =============================================================================

position_balance <- data_unedited %>%
  count(paragraph_position, paragraph_type) %>%
  group_by(paragraph_position) %>%
  mutate(share = n / sum(n)) %>%
  ungroup()
write_csv(position_balance, file.path(RESULTS_DIR, "position_balance.csv"))

balance_test <- chisq.test(table(data_unedited$paragraph_position, data_unedited$paragraph_type))
print(balance_test)

timing_by_position <- data_unedited %>%
  group_by(paragraph_position) %>%
  summarise(
    n = n(),
    median_seconds = median(paragraph_seconds),
    q25_seconds = quantile(paragraph_seconds, 0.25),
    q75_seconds = quantile(paragraph_seconds, 0.75),
    mean_item_sd = mean(item_sd, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(timing_by_position, file.path(RESULTS_DIR, "timing_by_position.csv"))

exclusion_counts <- tibble(
  check = c("speeder", "straightliner"),
  n_excluded = c(sum(data_robustness$speeder), sum(data_robustness$straightliner)),
  n_total = nrow(data_robustness),
  share_excluded = n_excluded / n_total,
  threshold = c(
    paste0("rater median seconds <= ", round(speeder_rater_threshold, 1),
           " or paragraph seconds < ", SPEEDER_PARAGRAPH_SECONDS),
    paste0("item SD <= ", round(straightliner_threshold, 2))
  )
)
write_csv(exclusion_counts, file.path(RESULTS_DIR, "exclusion_counts.csv"))

# =============================================================================
# ANALYSIS: BETA REGRESSIONS UNDER ALTERNATIVE SPECIFICATIONS
# =============================================================================

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

# Fits beta regression of outcome on paragraph type (+ optional covariates) and
# returns the writer-vs-model average marginal effect on the 0-100 scale
fit_beta_ame <- function(df, outcome, covariates = NULL, by = NULL) {
  df$y <- squeeze01(pmin(pmax(df[[outcome]] / 100, 0), 1))

  rhs <- paste(c("paragraph_type_", covariates, "(1 | rater_id)"), collapse = " + ")
  model <- glmmTMB(as.formula(paste("y ~", rhs)), data = df, family = beta_family(link = "logit"))

  avg_comparisons(
    model,
    variables = list(paragraph_type_ = "reference"),
    by = if (is.null(by)) TRUE else by,
    type = "response",
    re.form = NA
  ) %>%
    as_tibble() %>%
    transmute(
      across(any_of(by)),
      ame = estimate * 100,
      ame_low = conf.low * 100,
      ame_high = conf.high * 100,
      p = p.value,
      n = nrow(df)
    )
}

specifications <- list(
  full              = list(filter = function(d) d),
  position_1        = list(filter = function(d) filter(d, paragraph_position == 1)),
  positions_1_3     = list(filter = function(d) filter(d, paragraph_position <= 3)),
  positions_6_10    = list(filter = function(d) filter(d, paragraph_position >= 6)),
  previous_type     = list(filter = function(d) d, covariates = "previous_type"),
  no_speeders       = list(filter = function(d) filter(d, !speeder)),
  no_straightliners = list(filter = function(d) filter(d, !straightliner))
)

ame_results <- list()
position_results <- list()

for (attribute in rating_attributes) {
  print(paste("running robustness regressions for:", attribute))

  attribute_data <- data_robustness
  if (demo_mode) {
    attribute_data <- attribute_data %>% slice_sample(n = min(1000, nrow(attribute_data)))
  }

  for (spec_name in names(specifications)) {
    spec <- specifications[[spec_name]]
    ame_results[[length(ame_results) + 1]] <- fit_beta_ame(
      spec$filter(attribute_data), attribute, covariates = spec$covariates
    ) %>%
      mutate(attribute = attribute, specification = spec_name, .before = 1)
  }

  # Writer-vs-model AME at each position from a type x position interaction
  position_results[[length(position_results) + 1]] <- fit_beta_ame(
    attribute_data, attribute,
    covariates = "paragraph_type_:paragraph_position + paragraph_position",
    by = "paragraph_position"
  ) %>%
    mutate(attribute = attribute, .before = 1)
}

write_csv(bind_rows(ame_results), file.path(RESULTS_DIR, "ame_by_specification.csv"))
write_csv(bind_rows(position_results), file.path(RESULTS_DIR, "ame_by_position.csv"))

if (demo_mode) {
  message("Ran in demo mode on n=1000 samples per attribute.")
}
