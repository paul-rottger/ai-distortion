#!/usr/bin/env Rscript

# =============================================================================
# STUDY OVERVIEW: PARTICIPANT AND ASSIGNMENT SUMMARIES
#
# Summarizes participant counts, rating volume, and assignment structure across studies.
#
# - Reports participant totals for each phase and followup study dataset.
# - Reports phase 2 rating counts and unique paragraph coverage when available.
# - Reports phase 1 assignment breakdowns by study-specific experimental factors.
# - Prints overview tables to console for quick cross-study validation.
#
# =============================================================================

# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
})

source("./analysis/utils_r/demo_paths.R")

args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)

# ===== DATA IMPORTS ----
studies <- c(
  "main_phase_1",
  "main_phase_2",
  "followup_disclaimer_phase_1",
  "followup_mitigation_phase_1",
  "followup_mitigation_phase_2"
)

# ===== HELPER FUNCTIONS ----
describe_participants <- function(participant_path) {
  participants <- read_csv(file.path("./data", participant_path), show_col_types = FALSE)
  cat("Number of participants:", nrow(participants), "\n")
}

describe_phase2_ratings <- function(study_name) {
  annotations_path <- file.path("./data", study_name, "annotations.csv")

  if (!file.exists(annotations_path)) {
    cat("Number of ratings: annotations.csv not found\n")
    cat("Number of unique paragraphs rated: annotations.csv not found\n")
    return(invisible(NULL))
  }

  annotations <- read_csv(annotations_path, show_col_types = FALSE)
  cat("Number of ratings:", nrow(annotations), "\n")

  unique_paragraphs <- unique(
    annotations[c("writer_id", "proposition_id", "paragraph_type")]
  )

  cat("Number of unique paragraphs rated:", nrow(unique_paragraphs), "\n")
}

describe_phase1_assignments <- function(study_name) {
  responses_path <- file.path("./data", study_name, "proposition_responses.csv")

  if (!file.exists(responses_path)) {
    cat("Assignment breakdown: proposition_responses.csv not found\n")
    return(invisible(NULL))
  }

  responses <- read_csv(responses_path, show_col_types = FALSE)

  if (study_name == "main_phase_1") {
    assignment_counts <- responses %>%
      distinct(writer_id, proposition_id, model_name, model_input_condition) %>%
      count(model_name, model_input_condition, name = "n_pairs") %>%
      rename(
        model = model_name,
        input_condition = model_input_condition
      ) %>%
      arrange(model, input_condition)
  } else if (study_name == "followup_disclaimer_phase_1") {
    assignment_counts <- responses %>%
      distinct(writer_id, proposition_id, disclaimer_condition) %>%
      count(disclaimer_condition, name = "n_pairs") %>%
      arrange(disclaimer_condition)
  } else if (study_name == "followup_mitigation_phase_1") {
    assignment_counts <- responses %>%
      distinct(writer_id, proposition_id, model_name, model_mitigation_condition) %>%
      count(model_name, model_mitigation_condition, name = "n_pairs") %>%
      rename(
        model = model_name,
        mitigation_condition = model_mitigation_condition
      ) %>%
      arrange(model, mitigation_condition)
  } else {
    cat("Assignment breakdown: not applicable\n")
    return(invisible(NULL))
  }

  print(assignment_counts, n = Inf)
}

# ===== DESCRIBE PARTICIPANTS ----
for (study in studies) {
  cat("Study:", study, "\n")
  describe_participants(paste0(study, "/participants.csv"))
  cat("\n")
}

# ===== PHASE 1 ASSIGNMENT OVERVIEW ----
phase1_studies <- c(
  "main_phase_1",
  "followup_disclaimer_phase_1",
  "followup_mitigation_phase_1"
)

cat("Phase 1 assignment overview:\n")
for (study in phase1_studies) {
  cat("Study:", study, "\n")
  describe_phase1_assignments(study)
  cat("\n")
}

# ===== PHASE 2 RATINGS OVERVIEW ----
phase2_studies <- studies[str_detect(studies, "phase_2")]

cat("Phase 2 ratings overview:\n")
for (study in phase2_studies) {
  cat("Study:", study, "\n")
  describe_phase2_ratings(study)
  cat("\n")
}
