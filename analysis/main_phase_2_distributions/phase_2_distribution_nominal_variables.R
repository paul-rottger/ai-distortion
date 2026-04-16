# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
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

# ===== NOMINAL VARIABLES: CHI-SQUARED TEST + CRAMER'S V ----
cramers_v <- function(data, rating_attribute) {
  tab <- table(data$paragraph_type_, data[[rating_attribute]])
  chisq <- suppressWarnings(chisq.test(tab, correct = FALSE))
  n <- sum(tab)
  dims <- dim(tab)
  sqrt(as.numeric(chisq$statistic) / (n * (min(dims) - 1)))
}

bootstrap_cramers_v_ci <- function(data, rating_attribute,
                                   R = 1000, conf = 0.95) {
  v_boot <- replicate(R, {
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    cramers_v(data[idx, ], rating_attribute)
  })

  alpha <- (1 - conf) / 2
  quantile(v_boot, c(alpha, 1 - alpha), na.rm = TRUE)
}

run_nominal_test <- function(data, rating_attribute) {
  # Build contingency table
  tab <- table(data$paragraph_type_, data[[rating_attribute]])

  # Run chi-squared test
  chisq <- chisq.test(tab)

  # Compute Cramer's V
  n <- sum(tab)
  dims <- dim(tab)
  cramers_v <- sqrt(as.numeric(chisq$statistic) / (n * (min(dims) - 1)))

  # Bootstrap confidence interval for Cramer's V
  cramers_v_ci <- bootstrap_cramers_v_ci(data, rating_attribute)
  cramers_v_ci_low <- cramers_v_ci[1]
  cramers_v_ci_high <- cramers_v_ci[2]

  list(
    test = chisq,
    cramers_v = cramers_v,
    cramers_v_ci_low = cramers_v_ci_low,
    cramers_v_ci_high = cramers_v_ci_high
  )
}

# Function to go through all combinations of data splits and predictors for a given nominal variable
run_all_nominal_tests <- function(attribute) {
  print(paste("running nominal variable tests for:", attribute))

  for (data_split in DATA_SPLITS) {
    dir.create(
      file.path(RESULTS_DIR, data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    results <- run_nominal_test(get_split_data(data_split), attribute)

    write_csv(
      tibble(
        term = "paragraph_type_model",
        statistic = results$test$statistic,
        p_value = results$test$p.value,
        cramers_v = results$cramers_v,
        cramers_v_ci_low = results$cramers_v_ci_low,
        cramers_v_ci_high = results$cramers_v_ci_high
      ),
      file.path(RESULTS_DIR, data_split, paste0(attribute, "_by_type.csv"))
    )
  }
}

# Loop through all rating attributes
for (attr in nominal_vars) {
  run_all_nominal_tests(attr)
}