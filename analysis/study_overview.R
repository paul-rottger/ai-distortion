# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
})

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")

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

# ===== DESCRIBE PARTICIPANTS ----
for (study in studies) {
  cat("Study:", study, "\n")
  describe_participants(paste0(study, "/participants.csv"))
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
