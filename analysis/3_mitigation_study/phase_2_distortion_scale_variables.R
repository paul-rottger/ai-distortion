# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
  "./data/followup_mitigation_phase_2/annotations.csv",
  "./data/followup_mitigation_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      mutate(
        model_                = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
        input_condition_      = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)), ref = "writer"),
        mitigation_condition_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_mitigation_condition)), ref = "writer"),
        model_and_mitigation_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", paste(model_name, model_mitigation_condition, sep = "__"))), ref = "writer")
      )
  }
), envir = environment())

# ===== BETA REGRESSION SETUP for SCALE VARIABLES ----

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

# Beta regression function
fit_beta <- function(df, outcome, predictor, random) {
  # 0–100 -> proportion, then squeeze to (0,1)
  y <- df[[outcome]] / 100
  y <- pmin(pmax(y, 0), 1)
  y <- squeeze01(y)

  # Build model
  form <- as.formula(paste0("y ~ ", predictor, " + ", random))
  model <- glmmTMB(form, data = df, family = beta_family(link = "logit"))

  # Fixed effects
  tidy_fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p = p.value
    ) %>%
    select(term, odds_ratio, or_low, or_high, statistic, p)

  # Compute average marginal effects
  ame <- avg_comparisons(
    model,
    variables = setNames(list("reference"), predictor),
    type = "response",
    re.form = NA
  ) %>%
    as_tibble() %>%
    mutate(
      term = paste0(predictor, sub(" .*", "", contrast)),
      ame = estimate * 100,
      ame_low = conf.low * 100,
      ame_high = conf.high * 100,
    ) %>%
    select(term, ame, ame_low, ame_high)

  tidy_fixed <- tidy_fixed %>% left_join(ame, by = "term")

  list(model = model, tidy_fixed = tidy_fixed)
}

# ===== RUN SINGLE REGRESSION FOR DEBUGGING ----

# Select subset of data for debugging
data_small <- data_unedited %>% slice_sample(n = 1000)

# Run regression
results <- fit_beta(data_small,
  outcome = "writer_knowledge",
  predictor = "paragraph_type_",
  random = "(1 | rater_id)"
)
results$tidy_fixed

# ===== RUN ALL REGRESSIONS ----

# Function to go through all combinations of data splits and predictors for a given attribute
run_regressions <- function(attribute) {
  print(paste("running regressions for:", attribute))

  for (data_split in c("preferred", "edited", "unedited")) {
    for (predictor in list(
      c("model_and_mitigation_", "by_model_and_mitigation")
    )) {
      split_data <- switch(data_split,
        unedited = data_unedited,
        edited = data_edited,
        preferred = data_preferred
      )

      dir.create(
        paste0("./results/followup_mitigation_phase_2_distortion/", data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- fit_beta(split_data,
        outcome = attribute,
        predictor = predictor[1],
        random = "(1 | rater_id)"
      )
      write_csv(
        results$tidy_fixed,
        paste0("./results/followup_mitigation_phase_2_distortion/", data_split, "/", attribute, "_", predictor[2], ".csv")
      )
    }
  }
}

# Loop through all rating attributes
for (attr in rating_attributes) {
  run_regressions(attr)
}
