# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(mclogit)
  library(lme4)
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

empty_nominal_results <- function() {
  tibble(
    term = character(),
    target_level = character(),
    reference_level = character(),
    odds_ratio = numeric(),
    or_low = numeric(),
    or_high = numeric(),
    statistic = numeric(),
    p = numeric(),
    p_value = numeric()
  )
}

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
    return(empty_nominal_results())
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

    return(tibble(
      term = term_name,
      target_level = setdiff(levels(model_fit$model_df[[outcome]]), reference_level),
      reference_level = reference_level,
      odds_ratio = NA_real_,
      or_low = NA_real_,
      or_high = NA_real_,
      statistic = NA_real_,
      p = NA_real_,
      p_value = NA_real_
    ))
  }

  coef_table %>%
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
    )
}

fit_ova_logit <- function(df,
                          outcome,
                          predictor = "paragraph_type_",
                          predictor_level = "model",
                          random_effect = "rater_id") {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]]), !is.na(.data[[random_effect]])) %>%
    mutate(
      outcome_factor = droplevels(as.factor(.data[[outcome]])),
      predictor_factor = droplevels(as.factor(.data[[predictor]])),
      random_effect_factor = as.factor(.data[[random_effect]])
    )

  if (nrow(model_df) == 0 || !(predictor_level %in% levels(model_df$predictor_factor))) {
    return(empty_nominal_results())
  }

  target_levels <- levels(model_df$outcome_factor)

  map_dfr(target_levels, function(target_level) {
    binary_df <- model_df %>%
      mutate(target_flag = as.integer(outcome_factor == target_level))

    if (n_distinct(binary_df$target_flag) < 2 || nlevels(binary_df$predictor_factor) < 2) {
      return(tibble(
        term = paste0(predictor, predictor_level),
        target_level = target_level,
        reference_level = "all_other_levels",
        odds_ratio = NA_real_,
        or_low = NA_real_,
        or_high = NA_real_,
        statistic = NA_real_,
        p = NA_real_,
        p_value = NA_real_
      ))
    }

    fit <- tryCatch(
      suppressWarnings(
        glmer(
          target_flag ~ predictor_factor + (1 | random_effect_factor),
          data = binary_df,
          family = binomial(link = "logit")
        )
      ),
      error = function(e) NULL
    )

    if (is.null(fit)) {
      return(tibble(
        term = paste0(predictor, predictor_level),
        target_level = target_level,
        reference_level = "all_other_levels",
        odds_ratio = NA_real_,
        or_low = NA_real_,
        or_high = NA_real_,
        statistic = NA_real_,
        p = NA_real_,
        p_value = NA_real_
      ))
    }

    fit_summary <- coef(summary(fit))
    term_name <- paste0("predictor_factor", predictor_level)

    if (!(term_name %in% rownames(fit_summary))) {
      return(tibble(
        term = paste0(predictor, predictor_level),
        target_level = target_level,
        reference_level = "all_other_levels",
        odds_ratio = NA_real_,
        or_low = NA_real_,
        or_high = NA_real_,
        statistic = NA_real_,
        p = NA_real_,
        p_value = NA_real_
      ))
    }

    estimate <- fit_summary[term_name, "Estimate"]
    std_error <- fit_summary[term_name, "Std. Error"]
    statistic <- fit_summary[term_name, "z value"]
    p_value <- fit_summary[term_name, "Pr(>|z|)"]

    tibble(
      term = paste0(predictor, predictor_level),
      target_level = target_level,
      reference_level = "all_other_levels",
      odds_ratio = exp(estimate),
      or_low = exp(estimate - 1.96 * std_error),
      or_high = exp(estimate + 1.96 * std_error),
      statistic = statistic,
      p = p_value,
      p_value = p_value
    )
  })
}

run_nominal_regressions <- function(attribute) {
  print(paste("running multinomial logistic regression for:", attribute))

  for (data_split in c("preferred")) {
    split_data <- switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      )

    multinomial_results <- fit_multinomial_logit(
      split_data,
      outcome = attribute,
      predictor = "paragraph_type_",
      outcome_ref = reference_levels[[attribute]]
    )

    ova_results <- fit_ova_logit(
      split_data,
      outcome = attribute,
      predictor = "paragraph_type_",
      predictor_level = "model",
      random_effect = "rater_id"
    )

    dir.create(
      paste0("./results/main_phase_2_distortion/", data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    dir.create(
      paste0("./results/main_phase_2_distortion/", data_split, "/ova_logistic_results"),
      recursive = TRUE,
      showWarnings = FALSE
    )

    write_csv(
      multinomial_results,
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
    )

    write_csv(
      ova_results,
      paste0(
        "./results/main_phase_2_distortion/",
        data_split,
        "/ova_logistic_results/",
        attribute,
        "_by_type.csv"
      )
    )
  }
}

# loop through nominal variables and run regressions
for (attribute in nominal_vars) {
  run_nominal_regressions(attribute)
}
