# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
  library(ordinal)
  library(parallel)
})

source("./analysis/utils_r/variable_definitions.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== EXECUTION FLAGS ----
RUN_DEBUG_ONLY <- FALSE

# ===== DATA IMPORTS ----
data <- read_csv("./data/main_phase_2/annotations.csv", show_col_types = FALSE)
phase_1_preferences <- read_csv("./data/main_phase_1/proposition_responses.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    rater_id = as.factor(rater_id),
    writer_id = as.factor(writer_id),
    model_ = factor(ifelse(paragraph_type == "writer", "writer", model_name)),
    input_condition_ = factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)),
    across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
  )

# Set reference category for predictors
data$model_ <- relevel(data$model_, ref = "writer")
data$input_condition_ <- relevel(data$input_condition_, ref = "writer")

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
  distinct(writer_id, proposition_id)

data_preferred <- data_edited %>%
  anti_join(preferred_exclusions, by = c("writer_id", "proposition_id"))

rm(data, phase_1_preferences, preferred_exclusions)


# Create random data sample for debugging
data_small <- data_unedited %>%
  sample_n(1000)

# ===== ORDINAL LOGISTIC REGRESSION (BY TYPE) ----

fit_ordinal_logit <- function(df, outcome, predictor = "paragraph_type_", random = "(1 | rater_id)") {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
    mutate(
      rater_id = as.factor(rater_id)
    )

  if (predictor == "paragraph_type_") {
    model_df <- model_df %>%
      mutate(paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer"))
  }

  if (predictor == "model_") {
    model_df <- model_df %>%
      mutate(model_ = relevel(as.factor(model_), ref = "writer"))
  }

  if (predictor == "input_condition_") {
    model_df <- model_df %>%
      mutate(input_condition_ = relevel(as.factor(input_condition_), ref = "writer"))
  }

  # Drop "Other" only when fitting writer_education model
  if (outcome == "writer_education") {
    model_df <- model_df %>%
      filter(writer_education != "Other")
  }

  # Cumulative link mixed model (logit link)
  form <- as.formula(paste0(outcome, " ~ ", predictor, " + ", random))
  model <- clmm(
    form,
    data = model_df,
    link = "logit",
    Hess = TRUE,
    nAGQ = 0,
    control = clmm.control(maxIter = 200, gradTol = 1e-4)
  )

  coef_tbl <- as_tibble(summary(model)$coefficients, rownames = "term")

  # Keep only predictor term(s), not threshold/cutpoint parameters
  tidy_fixed <- coef_tbl %>%
    filter(str_starts(term, predictor)) %>%
    transmute(
      term = term,
      estimate = Estimate,
      std_error = `Std. Error`,
      statistic = `z value`,
      p = `Pr(>|z|)`,
      conf.low = estimate - 1.96 * std_error,
      conf.high = estimate + 1.96 * std_error,
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p_value = p
    ) %>%
    dplyr::select(term, odds_ratio, or_low, or_high, statistic, p, p_value)

  list(model = model, tidy_fixed = tidy_fixed)
}

# Debug run
debug_runs <- list(
  list(outcome = "writer_income", predictor = "paragraph_type_"),
  list(outcome = "writer_income", predictor = "model_"),
  list(outcome = "writer_income", predictor = "input_condition_"),
  list(outcome = "writer_education", predictor = "model_")
)

debug_results <- purrr::map(
  debug_runs,
  ~ {
    fit_ordinal_logit(
      data_small,
      outcome = .x$outcome,
      predictor = .x$predictor,
      random = "(1 | rater_id)"
    )$tidy_fixed
  }
)

names(debug_results) <- purrr::map_chr(
  debug_runs,
  ~ paste(.x$outcome, .x$predictor, sep = "__")
)

print(debug_results)

run_ordinal_regressions <- function(attribute) {
  print(paste("running ordinal logistic regression for:", attribute))

  for (data_split in c("unedited", "edited")) {
    for (predictor in list(
      #c("paragraph_type_", "by_type")
      c("model_", "by_model"),
      c("input_condition_", "by_input_condition")
    )) {
      split_data <- switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      )

      dir.create(
        paste0("./results/main_phase_2_distortion/", data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- fit_ordinal_logit(
        split_data,
        outcome = attribute,
        predictor = predictor[1],
        random = "(1 | rater_id)"
      )

      write_csv(
        results$tidy_fixed,
        paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_", predictor[2], ".csv")
      )
    }
  }
}

if (!RUN_DEBUG_ONLY) {
  # Loop through all ordinal attributes and predictors
  for (attribute in ordinal_vars) {
    run_ordinal_regressions(attribute)
  }
}
