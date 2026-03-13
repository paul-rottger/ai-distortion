# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(nnet)
})

# ===== PLOTTING DEFAULTS ----
font_add(family = "CMU Serif", regular = "~/Library/Fonts/cmunrm.ttf")
showtext_auto()
theme_set(theme_minimal(base_family = "CMU Serif", base_size = 14))

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/main_phase_2/annotations.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    rater_id = as.factor(rater_id),
    writer_id = as.factor(writer_id),
    proposition_id = as.factor(proposition_id),
    model_ = factor(ifelse(paragraph_type == "writer", "writer", model_name)),
    input_condition_ = factor(ifelse(paragraph_type == "writer", "writer", model_input_condition))
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
  filter(!(paragraph_type == "model" & any(paragraph_type == "edited"))) %>%
  mutate(
    paragraph_type = if_else(paragraph_type == "edited", "model", paragraph_type),
    paragraph_type_ = as.factor(paragraph_type),
    paragraph_type_ = relevel(paragraph_type_, ref = "writer")
  ) %>%
  ungroup()

rm(data)

# ===== NOMINAL VARIABLES / REFERENCE CATEGORIES ----
nominal_vars <- c(
  "writer_race",
  "writer_gender",
  "writer_politicalParty",
  "writer_politicalIdeology"
)

reference_levels <- c(
  writer_race = "White",
  writer_gender = "Female",
  writer_politicalParty = "Labour",
  writer_politicalIdeology = "Centrist"
)

# ===== MULTINOMIAL LOGISTIC REGRESSION (BY TYPE) ----
fit_multinomial_logit <- function(df, outcome, predictor = "paragraph_type_", outcome_ref = NULL) {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
    mutate(
      paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer")
    )

  if (nrow(model_df) == 0) {
    return(tibble())
  }

  model_df[[outcome]] <- as.factor(model_df[[outcome]])
  model_df[[outcome]] <- droplevels(model_df[[outcome]])

  if (!is.null(outcome_ref) && outcome_ref %in% levels(model_df[[outcome]])) {
    model_df[[outcome]] <- relevel(model_df[[outcome]], ref = outcome_ref)
  }

  if (nlevels(model_df[[outcome]]) < 2) {
    return(tibble())
  }

  form <- as.formula(paste0(outcome, " ~ ", predictor))

  fit <- tryCatch(
    suppressWarnings(nnet::multinom(form, data = model_df, trace = FALSE)),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(tibble())
  }

  fit_summary <- summary(fit)
  coefs <- fit_summary$coefficients
  ses <- fit_summary$standard.errors

  if (is.null(dim(coefs))) {
    coefs <- matrix(coefs, nrow = 1)
    ses <- matrix(ses, nrow = 1)
    colnames(coefs) <- names(fit_summary$coefficients)
    colnames(ses) <- names(fit_summary$standard.errors)
    rownames(coefs) <- levels(model_df[[outcome]])[2]
    rownames(ses) <- levels(model_df[[outcome]])[2]
  }

  target_levels <- rownames(coefs)
  if (is.null(target_levels)) {
    target_levels <- setdiff(levels(model_df[[outcome]]), levels(model_df[[outcome]])[1])
  }

  term_name <- paste0(predictor, "model")

  if (!(term_name %in% colnames(coefs))) {
    return(tibble(
      term = term_name,
      target_level = target_levels,
      reference_level = levels(model_df[[outcome]])[1],
      odds_ratio = NA_real_,
      or_low = NA_real_,
      or_high = NA_real_,
      statistic = NA_real_,
      p = NA_real_,
      p_value = NA_real_
    ))
  }

  estimate <- coefs[, term_name]
  std_error <- ses[, term_name]
  statistic <- estimate / std_error
  p_value <- 2 * pnorm(abs(statistic), lower.tail = FALSE)

  tibble(
    term = term_name,
    target_level = target_levels,
    reference_level = levels(model_df[[outcome]])[1],
    odds_ratio = exp(estimate),
    or_low = exp(estimate - 1.96 * std_error),
    or_high = exp(estimate + 1.96 * std_error),
    statistic = statistic,
    p = p_value,
    p_value = p_value
  )
}

# Debug run
data_small <- data_unedited %>% sample_n(2000)
fit_multinomial_logit(data_small, "writer_gender", outcome_ref = reference_levels[["writer_gender"]])

run_nominal_regressions <- function(attribute) {
  print(paste("running multinomial logistic regression for:", attribute))

  # Matching ordinal script approach: edited only, by_type only
  for (data_split in c("edited")) {
    results <- fit_multinomial_logit(
      if (data_split == "unedited") data_unedited else data_edited,
      outcome = attribute,
      predictor = "paragraph_type_",
      outcome_ref = reference_levels[[attribute]]
    )

    write_csv(
      results,
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
    )
  }
}

# Loop through all nominal attributes for multinomial logistic regression
for (attr in nominal_vars) {
  run_nominal_regressions(attr)
}
