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
  filter(paragraph_type %in% c("writer", "model")) %>%
  mutate(
    paragraph_type_ = factor(paragraph_type),
    writer_age_binned = factor(writer_age_binned, levels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"), ordered = TRUE),
    writer_english_first = factor(writer_english_first, levels = c("No", "Yes"), ordered = TRUE),
    writer_english_skills = factor(writer_english_skills, levels = c("Basic", "Intermediate", "Advanced", "Expert"), ordered = TRUE),
    writer_education = factor(writer_education, levels = c("GCSEs or equivalent", "A-levels or equivalent", "Vocational qualification", "Undergraduate degree", "Postgraduate degree (Master's)", "Doctorate (PhD)", "Other"), ordered = TRUE),
    writer_income = factor(writer_income, levels = c("Under £15,000", "£15,000-£24,999", "£25,000-£34,999", "£35,000-£49,999", "£50,000-£74,999", "£75,000-£99,999", "£100,000+"), ordered = TRUE),
  )

# drop "other" category for writer_education -almost equal proportion across groups and breaks ordinality
data <- data %>%
  filter(writer_education != "Other")

# Set reference category for predictors
data$paragraph_type_ <- relevel(data$paragraph_type_, ref = "model")

# Create random data sample for debugging
data_small <- data %>%
  sample_n(5000)

# ===== NOMINAL VARIABLES: CHI-SQUARED TEST + CRAMER'S V ----
run_chisq <- function(data, rating_attribute) {
  # Build contingency table
  tab <- table(data$paragraph_type_, data[[rating_attribute]])

  # Run chi-squared test
  chisq <- chisq.test(tab)

  # Compute Cramer's V
  n <- sum(tab)
  dims <- dim(tab)
  cramers_v <- sqrt(as.numeric(chisq$statistic) / (n * (min(dims) - 1)))

  list(
    table = tab,
    test = chisq,
    cramers_v = cramers_v
  )
}

nominal_vars <- c(
  "writer_gender",
  "writer_race",
  "writer_politicalParty",
  "writer_politicalIdeology"
)

results <- lapply(nominal_vars, function(var) {
  run_chisq(data, var)
})

# Print results
for (i in seq_along(nominal_vars)) {
  cat("Nominal Variable:", nominal_vars[i], "\n")
  # print(results[[i]]$table)
  cat("p-value:", results[[i]]$test$p.value, "\n")
  cat("Cramer's V:", round(results[[i]]$cramers_v, 3), "\n")
  cat("\n")
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

run_ordinal_test(data, "writer_income")

ordinal_vars <- c(
  "writer_income",
  "writer_age_binned",
  "writer_english_first",
  "writer_english_skills",
  "writer_education"
)

ordinal_results <- lapply(ordinal_vars, function(var) {
  run_ordinal_test(data, var)
})

# Print results
for (i in seq_along(ordinal_vars)) {
  cat("Ordinal Variable:", ordinal_vars[i], "\n")
  cat("p-value:", ordinal_results[[i]]$test$p.value, "\n")
  cat("Cliff's delta:", round(ordinal_results[[i]]$cliffs_delta, 3), "[", round(ordinal_results[[i]]$cliffs_delta_ci_low, 3), ",", round(ordinal_results[[i]]$cliffs_delta_ci_high, 3), "]\n")
  cat("\n")
}
