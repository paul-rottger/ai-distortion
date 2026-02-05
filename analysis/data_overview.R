# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
  
})

# ===== PLOTTING DEFAULTS ----
font_add(family = "CMU Serif", regular = "~/Library/Fonts/cmunrm.ttf")
showtext_auto()
theme_set(theme_minimal(base_family = "CMU Serif", base_size = 14))

# ===== RANDOM SEED ----
set.seed(123)

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

# ===== DESCRIBE PARTICIPANTS ----
for (study in studies) {
  cat("Study:", study, "\n")
  describe_participants(paste0(study, "/participants.csv"))
  cat("\n")
}
