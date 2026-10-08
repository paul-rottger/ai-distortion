#!/usr/bin/env Rscript

# =============================================================================
# FOLLOWUP MITIGATION STUDY - PHASE 2 INTER-RATER AGREEMENT
#
# Quantifies how consistently readers rate the same paragraph on each of the
# 29 annotation attributes (scale, ordinal, and nominal).
#
# - Units are paragraphs (writer x proposition x paragraph type x model x
#   input condition x mitigation condition); each paragraph is rated by ~5
#   different raters.
# - Computes Krippendorff's alpha (interval / ordinal / nominal metric by
#   variable type), which handles the sparse crossed rater design.
# - For scale attributes, complements alpha with the mean within-paragraph SD
#   (in rating points).
# - Reports metrics for all paragraphs and separately by paragraph type.
# - Writes alpha.csv and scale_mean_sd.csv to
#   results/followup_mitigation_phase_2_agreement/
#   (rows: attributes; columns: overall, writer, model, edited).
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/rater_agreement.R")

# Set random seed for reproducibility
set.seed(123)

# Parse command-line flags
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "followup_mitigation_phase_2_agreement")

UNIT_COLUMNS <- c(
  "writer_id", "proposition_id", "paragraph_type", "model_name",
  "model_input_condition", "model_mitigation_condition"
)

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

data <- read_csv("./data/followup_mitigation_phase_2/annotations.csv", show_col_types = FALSE) %>%
  unite("unit_id", all_of(UNIT_COLUMNS), remove = FALSE)

if (demo_mode) {
  message("Running in demo mode on n=1000 sampled paragraphs.")
  demo_units <- sample(unique(data$unit_id), min(1000, n_distinct(data$unit_id)))
  data <- data %>% filter(unit_id %in% demo_units)
}

# =============================================================================
# ANALYSIS: INTER-RATER AGREEMENT
# =============================================================================

run_rater_agreement(data, RESULTS_DIR)
