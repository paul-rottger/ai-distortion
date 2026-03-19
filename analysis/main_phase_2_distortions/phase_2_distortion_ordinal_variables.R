# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(effsize)
  library(ordinal)
  library(parallel)
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
    model_ = factor(ifelse(paragraph_type == "writer", "writer", model_name)),
    input_condition_ = factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)),
    writer_age_binned = factor(writer_age_binned, levels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"), ordered = TRUE),
    writer_english_first = factor(writer_english_first, levels = c("No", "Yes"), ordered = TRUE),
    writer_english_skills = factor(writer_english_skills, levels = c("Basic", "Intermediate", "Advanced", "Expert"), ordered = TRUE),
    writer_education = factor(writer_education, levels = c("GCSEs or equivalent", "A-levels or equivalent", "Vocational qualification", "Undergraduate degree", "Postgraduate degree (Master's)", "Doctorate (PhD)", "Other"), ordered = TRUE),
    writer_income = factor(writer_income, levels = c("Under £15,000", "£15,000-£24,999", "£25,000-£34,999", "£35,000-£49,999", "£50,000-£74,999", "£75,000-£99,999", "£100,000+"), ordered = TRUE)
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

ordinal_vars <- c(
  "writer_education",
  "writer_english_skills",
  "writer_income",
  "writer_age_binned",
  "writer_english_first"
)

# Create random data sample for debugging
data_small <- data_unedited %>%
  sample_n(1000)

# ===== ORDINAL LOGISTIC REGRESSION (BY TYPE) ----

fit_ordinal_logit <- function(df, outcome, predictor = "paragraph_type_", random = "(1 | rater_id)") {
  model_df <- df %>%
    filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
    mutate(
      rater_id = as.factor(rater_id),
      paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer")
    )

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
    select(term, odds_ratio, or_low, or_high, statistic, p, p_value)

  list(model = model, tidy_fixed = tidy_fixed)
}

# Debug run
purrr::map(
  c("writer_income", "writer_education"),
  ~ fit_ordinal_logit(data_small, .x)$tidy_fixed
)

run_ordinal_regressions <- function(attribute) {
  print(paste("running ordinal logistic regression for:", attribute))

  for (data_split in c("preferred")) {
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
      predictor = "paragraph_type_",
      random = "(1 | rater_id)"
    )

    write_csv(
      results$tidy_fixed,
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
    )
  }
}

# Loop through all ordinal attributes for by-type ordinal logistic regression (parallel)
n_cores <- max(1, parallel::detectCores() - 1)
parallel::mclapply(ordinal_vars, run_ordinal_regressions, mc.cores = n_cores)
