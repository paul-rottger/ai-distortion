# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(mclogit)
  library(lme4)
})

source("./analysis/utils_r/variable_definitions.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
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

relevel_if_present <- function(x, ref) {
  if (ref %in% levels(x)) {
    return(relevel(x, ref = ref))
  }

  x
}

prepare_predictor <- function(model_df, predictor) {
  model_df[[predictor]] <- droplevels(as.factor(model_df[[predictor]]))
  model_df[[predictor]] <- relevel_if_present(model_df[[predictor]], ref = "writer")
  model_df
}

get_multinomial_random_effects <- function(attribute) {
  if (attribute %in% c("writer_politicalParty", "writer_politicalIdeology")) {
    return(NULL)
  }

  ~ 1 | rater_id
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
      rater_id = as.factor(rater_id)
    )

  model_df <- prepare_predictor(model_df, predictor)

  if (nrow(model_df) == 0 || nlevels(model_df[[predictor]]) < 2) {
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

  fit_args <- list(
    formula = form,
    data = model_df,
    estimator = "ML"
  )

  if (!is.null(random_effects)) {
    fit_args$random <- random_effects
  }

  fit <- suppressWarnings(do.call(mclogit::mblogit, fit_args))

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

  coef_table <- as_tibble(coefs, rownames = "row_id") %>%
    separate(row_id, into = c("target_level", "term"), sep = "~", remove = TRUE) %>%
    filter(str_starts(term, predictor))

  if (nrow(coef_table) == 0) {
    reference_level <- levels(model_fit$model_df[[outcome]])[1]
    predictor_levels <- levels(model_fit$model_df[[predictor]])
    predictor_terms <- paste0(predictor, predictor_levels[predictor_levels != predictor_levels[1]])
    target_levels <- setdiff(levels(model_fit$model_df[[outcome]]), reference_level)

    if (length(predictor_terms) == 0 || length(target_levels) == 0) {
      return(empty_nominal_results())
    }

    return(expand_grid(
      term = predictor_terms,
      target_level = target_levels
    ) %>%
      mutate(
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
                          predictor_ref = "writer",
                          random_effect = "rater_id") {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]]), !is.na(.data[[random_effect]])) %>%
    mutate(
      outcome_factor = droplevels(as.factor(.data[[outcome]])),
      predictor_factor = droplevels(as.factor(.data[[predictor]])),
      random_effect_factor = as.factor(.data[[random_effect]])
    )

  model_df$predictor_factor <- relevel_if_present(model_df$predictor_factor, ref = predictor_ref)

  if (nrow(model_df) == 0 || nlevels(model_df$predictor_factor) < 2) {
    return(empty_nominal_results())
  }

  target_levels <- levels(model_df$outcome_factor)
  comparison_levels <- setdiff(levels(model_df$predictor_factor), predictor_ref)

  if (length(comparison_levels) == 0) {
    return(empty_nominal_results())
  }

  map_dfr(target_levels, function(target_level) {
    binary_df <- model_df %>%
      mutate(target_flag = as.integer(outcome_factor == target_level))

    if (n_distinct(binary_df$target_flag) < 2 || nlevels(binary_df$predictor_factor) < 2) {
      return(map_dfr(comparison_levels, function(predictor_level) {
        tibble(
          term = paste0(predictor, predictor_level),
          target_level = target_level,
          reference_level = "all_other_levels",
          odds_ratio = NA_real_,
          or_low = NA_real_,
          or_high = NA_real_,
          statistic = NA_real_,
          p = NA_real_,
          p_value = NA_real_
        )
      }))
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
      return(map_dfr(comparison_levels, function(predictor_level) {
        tibble(
          term = paste0(predictor, predictor_level),
          target_level = target_level,
          reference_level = "all_other_levels",
          odds_ratio = NA_real_,
          or_low = NA_real_,
          or_high = NA_real_,
          statistic = NA_real_,
          p = NA_real_,
          p_value = NA_real_
        )
      }))
    }

    fit_summary <- coef(summary(fit))
    map_dfr(comparison_levels, function(predictor_level) {
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
  })
}

run_nominal_regressions <- function(attribute) {
  print(paste("running multinomial logistic regression for:", attribute))

  multinomial_random_effects <- get_multinomial_random_effects(attribute)

  if (is.null(multinomial_random_effects)) {
    print(paste("using fixed-effects multinomial fallback for:", attribute))
  }

  for (data_split in c("preferred", "edited", "unedited")) {
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

    dir.create(
      paste0("./results/main_phase_2_distortion/", data_split, "/ova_logistic_results"),
      recursive = TRUE,
      showWarnings = FALSE
    )

    for (predictor in list(
      #c("paragraph_type_", "by_type"),
      c("model_", "by_model"),
      c("input_condition_", "by_input")
    )) {
      multinomial_results <- fit_multinomial_logit(
        split_data,
        predictor = predictor[1],
        outcome = attribute,
        random_effects = multinomial_random_effects,
        outcome_ref = reference_levels[[attribute]]
      )

      ova_results <- fit_ova_logit(
        split_data,
        outcome = attribute,
        predictor = predictor[1],
        predictor_ref = "writer",
        random_effect = "rater_id"
      )

      write_csv(
        multinomial_results,
        paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_", predictor[2], ".csv")
      )

      write_csv(
        ova_results,
        paste0(
          "./results/main_phase_2_distortion/",
          data_split,
          "/ova_logistic_results/",
          attribute,
          "_",
          predictor[2],
          ".csv"
        )
      )
    }
  }
}

# loop through nominal variables and run regressions
for (attribute in nominal_vars) {
  run_nominal_regressions(attribute)
}
