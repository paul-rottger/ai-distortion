# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(mclogit)
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
phase_1_preferences <- read_csv("./data/main_phase_1/proposition_responses.csv", show_col_types = FALSE)

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

# ===== NOMINAL VARIABLES / REFERENCE CATEGORIES ----
nominal_vars <- c(
  "writer_race",
  "writer_gender",
  "writer_politicalParty",
  "writer_politicalIdeology"
)

reference_levels <- c(
  writer_race = "White",
  writer_gender = "Male",
  writer_politicalParty = "Labour",
  writer_politicalIdeology = "Centrist"
)

# ===== MULTINOMIAL LOGISTIC REGRESSION (BY TYPE) ----
fit_multinomial_logit_model <- function(df,
                                        outcome,
                                        predictor = "paragraph_type_",
                                        random_effects = ~ 1 | rater_id,
                                        outcome_ref = NULL) {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
    mutate(
      rater_id = as.factor(rater_id),
      paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer")
    )

  if (nrow(model_df) == 0) {
    return(NULL)
  }

  model_df[[outcome]] <- as.factor(model_df[[outcome]])
  model_df[[outcome]] <- droplevels(model_df[[outcome]])

  if (!is.null(outcome_ref) && outcome_ref %in% levels(model_df[[outcome]])) {
    model_df[[outcome]] <- relevel(model_df[[outcome]], ref = outcome_ref)
  }

  if (nlevels(model_df[[outcome]]) < 2) {
    return(NULL)
  }

  form <- as.formula(paste0(outcome, " ~ ", predictor))

  fit <- tryCatch(
    suppressWarnings(
      mclogit::mblogit(
        form,
        random = random_effects,
        data = model_df,
        estimator = "ML"
      )
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(NULL)
  }

  list(
    model = fit,
    model_df = model_df
  )
}

calculate_total_variation_distance <- function(model_fit,
                                               predictor = "paragraph_type_") {
  predictor_levels <- levels(model_fit$model_df[[predictor]])

  if (!all(c("writer", "model") %in% predictor_levels)) {
    return(NA_real_)
  }

  newdata <- setNames(
    data.frame(factor(c("writer", "model"), levels = predictor_levels)),
    predictor
  )

  if ("rater_id" %in% names(model_fit$model_df)) {
    newdata$rater_id <- model_fit$model_df$rater_id[[1]]
  }

  predicted_probs <- tryCatch(
    predict(model_fit$model, newdata = newdata, type = "response"),
    error = function(e) NULL
  )

  if (is.null(predicted_probs) || nrow(predicted_probs) != 2) {
    return(NA_real_)
  }

  0.5 * sum(abs(predicted_probs[2, ] - predicted_probs[1, ]))
}

fit_multinomial_logit <- function(df,
                                  outcome,
                                  predictor = "paragraph_type_",
                                  random_effects = ~ 1 | rater_id,
                                  outcome_ref = NULL) {
  model_fit <- fit_multinomial_logit_model(
    df = df,
    outcome = outcome,
    predictor = predictor,
    random_effects = random_effects,
    outcome_ref = outcome_ref
  )

  if (is.null(model_fit)) {
    return(list(results = tibble(), total_variation_distance = NA_real_))
  }

  fit_summary <- summary(model_fit$model)
  coefs <- fit_summary$coefficients

  if (is.null(dim(coefs))) {
    coefs <- matrix(coefs, nrow = 1)
    colnames(coefs) <- names(fit_summary$coefficients)
    rownames(coefs) <- paste0(levels(model_fit$model_df[[outcome]])[2], "~(Intercept)")
  }

  target_levels <- rownames(coefs)
  if (is.null(target_levels)) {
    target_levels <- setdiff(
      levels(model_fit$model_df[[outcome]]),
      levels(model_fit$model_df[[outcome]])[1]
    )
  }

  term_name <- paste0(predictor, "model")
  coef_table <- as_tibble(coefs, rownames = "row_id") %>%
    separate(row_id, into = c("target_level", "term"), sep = "~", remove = TRUE) %>%
    filter(term == term_name)

  if (nrow(coef_table) == 0) {
    reference_level <- levels(model_fit$model_df[[outcome]])[1]

    return(list(
      results = tibble(
        term = term_name,
        target_level = setdiff(levels(model_fit$model_df[[outcome]]), reference_level),
        reference_level = reference_level,
        odds_ratio = NA_real_,
        or_low = NA_real_,
        or_high = NA_real_,
        statistic = NA_real_,
        p = NA_real_,
        p_value = NA_real_
      ),
      total_variation_distance = calculate_total_variation_distance(model_fit, predictor = predictor)
    ))
  }

  list(
    results = coef_table %>%
      transmute(
        term = term,
        target_level = target_level,
        reference_level = levels(model_fit$model_df[[outcome]])[1],
        odds_ratio = exp(Estimate),
        or_low = exp(Estimate - 1.96 * `Std. Error`),
        or_high = exp(Estimate + 1.96 * `Std. Error`),
        statistic = `z value`,
        p = `Pr(>|z|)`,
        p_value = `Pr(>|z|)`
      ),
    total_variation_distance = calculate_total_variation_distance(model_fit, predictor = predictor)
  )
}

# Debug run
data_small <- data_unedited %>% sample_n(2000)
debug_results <- fit_multinomial_logit(data_small, "writer_gender", outcome_ref = reference_levels[["writer_gender"]])
print(debug_results$results)
print(tibble(
  outcome = "writer_gender",
  total_variation_distance = debug_results$total_variation_distance
))

run_nominal_regressions <- function(attribute) {
  print(paste("running multinomial logistic regression for:", attribute))

  for (data_split in c("preferred")) {
    results <- fit_multinomial_logit(
      switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      ),
      outcome = attribute,
      predictor = "paragraph_type_",
      outcome_ref = reference_levels[[attribute]]
    )

    print(tibble(
      outcome = attribute,
      data_split = data_split,
      total_variation_distance = results$total_variation_distance
    ))

    dir.create(
      paste0("./results/main_phase_2_distortion/", data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    write_csv(
      results$results,
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
    )
  }
}

# loop through nominal variables and run regressions
for (attribute in nominal_vars) {
  run_nominal_regressions(attribute)
}
