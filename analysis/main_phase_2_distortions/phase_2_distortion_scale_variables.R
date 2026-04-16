# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
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
    model_ = factor(ifelse(
      paragraph_type == "writer", "writer", model_name
    )),
    input_condition_ = factor(
      ifelse(paragraph_type == "writer", "writer", model_input_condition)
    ),
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

  for (data_split in c("preferred","edited","unedited")) {
    for (predictor in list(
      #c("paragraph_type_", "by_type")
      c("model_", "by_model"),
      c("input_condition_", "by_input")
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

      results <- fit_beta(split_data,
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

# Loop through all rating attributes
for (attr in rating_attributes) {
  run_regressions(attr)
}
