# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(effsize)
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

rm(data)

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

  for (data_split in c("unedited", "edited")) {
    results <- run_nominal_test(if (data_split == "unedited") data_unedited else data_edited, attribute)
    write_csv(
      tibble(
        term = "paragraph_type_model",
        statistic = results$test$statistic,
        p_value = results$test$p.value,
        cramers_v = results$cramers_v,
        cramers_v_ci_low = results$cramers_v_ci_low,
        cramers_v_ci_high = results$cramers_v_ci_high
      ),
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
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


# ===== ORDINAL VARIABLES: MANN–WHITNEY U + CLIFF'S DELTA ----

run_ordinal_test <- function(data, ordinal_var) {
  # Convert to numeric for testing
  x <- as.numeric(data[[ordinal_var]])
  g <- as.numeric(data$paragraph_type_)

  # Mann–Whitney U test
  mw <- wilcox.test(x ~ g, exact = FALSE)

  # Cliff's delta
  cd <- cliff.delta(x ~ g)

  list(
    test = mw,
    cliffs_delta = cd$estimate,
    cliffs_delta_ci_low = cd$conf.int[1],
    cliffs_delta_ci_high = cd$conf.int[2]
  )
}

run_ordinal_test(data_small, "writer_income")

# Function to go through all combinations of data splits and predictors for a given nominal variable
run_all_ordinal_tests <- function(attribute) {
  print(paste("running ordinal variable tests for:", attribute))

  for (data_split in c("unedited", "edited")) {
    results <- run_ordinal_test(if (data_split == "unedited") data_unedited else data_edited, attribute)
    write_csv(
      tibble(
        term = "paragraph_type_model",
        statistic = results$test$statistic,
        p_value = results$test$p.value,
        cliffs_delta = -results$cliffs_delta, # negate to reflect direction of effect (model vs writer)
        cliffs_delta_ci_high = -results$cliffs_delta_ci_low,
        cliffs_delta_ci_low = -results$cliffs_delta_ci_high
      ),
      paste0("./results/main_phase_2_distortion/", data_split, "/", attribute, "_by_type.csv")
    )
  }
}

ordinal_vars <- c(
  "writer_income",
  "writer_age_binned",
  "writer_english_first",
  "writer_english_skills",
  "writer_education"
)

# Loop through all rating attributes
for (attr in ordinal_vars) {
  run_all_ordinal_tests(attr)
}
