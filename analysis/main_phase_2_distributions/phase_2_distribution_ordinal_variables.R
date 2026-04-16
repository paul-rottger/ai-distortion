# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
  library(ordinal)
})

source("./analysis/utils_r/variable_definitions.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- "./results/main_phase_2_distribution"
DATA_SPLITS <- c("unedited", "edited", "preferred")

# ===== DATA IMPORTS ----
data <- read_csv("./data/main_phase_2/annotations.csv", show_col_types = FALSE)
phase_1_preferences <- read_csv("./data/main_phase_1/proposition_responses.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    rater_id = as.factor(rater_id),
    writer_id = as.factor(writer_id),
    proposition_id = as.factor(proposition_id),
    paragraph_type_ = factor(paragraph_type),
    across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
  )

# Drop "other" category for writer_education - almost equal proportion <1% across groups, dropping fixes ordinality
data <- data %>%
  filter(writer_education != "Other")

# Create unedited and edited subsets of data for later analyses
data_unedited <- data %>%
  filter(paragraph_type %in% c("writer", "model")) %>%
  mutate(
    paragraph_type_ = as.factor(paragraph_type),
    paragraph_type_ = relevel(paragraph_type_, ref = "writer")
  )

data_edited <- data %>%
  group_by(writer_id, proposition_id) %>%
  filter(!(paragraph_type == "model" &
    any(paragraph_type == "edited"))) %>%
  mutate(
    paragraph_type = if_else(paragraph_type == "edited",
      "model",
      paragraph_type
    ),
    paragraph_type_ = as.factor(paragraph_type),
    paragraph_type_ = relevel(paragraph_type_, ref = "writer")
  ) %>%
  ungroup()

preferred_exclusions <- phase_1_preferences %>%
  filter(writer_preference == "original") %>%
  mutate(
    writer_id = as.factor(writer_id),
    proposition_id = as.factor(proposition_id)
  ) %>%
  distinct(writer_id, proposition_id)

data_preferred <- data_edited %>%
  anti_join(preferred_exclusions, by = c("writer_id", "proposition_id"))

rm(data, phase_1_preferences, preferred_exclusions)

get_split_data <- function(data_split) {
  switch(data_split,
    unedited = data_unedited,
    edited = data_edited,
    preferred = data_preferred
  )
}

# Create random data sample for debugging
data_small <- data_unedited %>%
  sample_n(5000)


# ===== ORDINAL VARIABLES: MANN–WHITNEY U + CLIFF'S DELTA ----

run_ordinal_test <- function(data, ordinal_var) {
  # Convert to numeric for testing
  x <- as.numeric(data[[ordinal_var]])
  g <- as.numeric(data$paragraph_type_)

  # Mann–Whitney U test
  mw <- wilcox.test(x ~ g, exact = FALSE)

  # Cliff's delta
  cd <- cliff.delta(x ~ g)

  list(
    test = mw,
    cliffs_delta = cd$estimate,
    cliffs_delta_ci_low = cd$conf.int[1],
    cliffs_delta_ci_high = cd$conf.int[2]
  )
}

run_ordinal_test(data_small, "writer_income")

# Function to go through all combinations of data splits and predictors for a given ordinal variable
run_all_ordinal_tests <- function(attribute) {
  print(paste("running ordinal variable tests for:", attribute))

  for (data_split in DATA_SPLITS) {
    dir.create(
      file.path(RESULTS_DIR, data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    results <- run_ordinal_test(get_split_data(data_split), attribute)

    write_csv(
      tibble(
        term = "paragraph_type_model",
        statistic = results$test$statistic,
        p_value = results$test$p.value,
        cliffs_delta = -results$cliffs_delta, # negate to reflect direction of effect (model vs writer)
        cliffs_delta_ci_high = -results$cliffs_delta_ci_low,
        cliffs_delta_ci_low = -results$cliffs_delta_ci_high
      ),
      file.path(RESULTS_DIR, data_split, paste0(attribute, "_by_type.csv"))
    )
  }
}

# Loop through all rating attributes
for (attr in ordinal_vars) {
  run_all_ordinal_tests(attr)
}
