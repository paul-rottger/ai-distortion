# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
  library(ordinal)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- "./results/main_phase_2_distribution"
DATA_SPLITS <- c("unedited", "edited", "preferred")

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      # Drop "other" category for writer_education - almost equal proportion <1% across groups, dropping fixes ordinality
      filter(writer_education != "Other") %>%
      mutate(across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE)))
  }
), envir = environment())

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
