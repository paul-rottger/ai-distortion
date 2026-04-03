# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(ordinal)
})

# ===== PLOTTING DEFAULTS ----
font_add(family = "CMU Serif", regular = "~/Library/Fonts/cmunrm.ttf")
showtext_auto()
theme_set(theme_minimal(base_family = "CMU Serif", base_size = 14))

# ===== RANDOM SEED ----
set.seed(123)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- "./results/main_phase_2_distribution"
DATA_SPLITS <- c("unedited", "edited", "preferred")

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
    paragraph_type_ = factor(paragraph_type),
    writer_age_binned = factor(writer_age_binned, levels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"), ordered = TRUE),
    writer_english_first = factor(writer_english_first, levels = c("No", "Yes"), ordered = TRUE),
    writer_english_skills = factor(writer_english_skills, levels = c("Basic", "Intermediate", "Advanced", "Expert"), ordered = TRUE),
    writer_education = factor(writer_education, levels = c("GCSEs or equivalent", "A-levels or equivalent", "Vocational qualification", "Undergraduate degree", "Postgraduate degree (Master's)", "Doctorate (PhD)", "Other"), ordered = TRUE),
    writer_income = factor(writer_income, levels = c("Under £15,000", "£15,000-£24,999", "£25,000-£34,999", "£35,000-£49,999", "£50,000-£74,999", "£75,000-£99,999", "£100,000+"), ordered = TRUE)
  )

# Drop "other" category for writer_education - almost equal proportion <1% across groups, dropping fixes ordinality
data <- data %>%
  filter(writer_education != "Other")

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
  mutate(
    writer_id = as.factor(writer_id),
    proposition_id = as.factor(proposition_id)
  ) %>%
  distinct(writer_id, proposition_id)

data_preferred <- data_edited %>%
  anti_join(preferred_exclusions, by = c("writer_id", "proposition_id"))

rm(data, phase_1_preferences, preferred_exclusions)

get_split_data <- function(data_split) {
  switch(data_split,
    unedited = data_unedited,
    edited = data_edited,
    preferred = data_preferred
  )
}

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

nominal_vars <- c(
  "writer_gender",
  "writer_race",
  "writer_politicalParty",
  "writer_politicalIdeology"
)

# Loop through all rating attributes
for (attr in nominal_vars) {
  run_all_nominal_tests(attr)
}