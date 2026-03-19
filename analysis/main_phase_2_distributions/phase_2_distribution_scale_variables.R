# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
})

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORT ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/main_phase_2/annotations.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    paragraph_type_ = factor(paragraph_type)
  )

data$paragraph_type_ <- relevel(data$paragraph_type_, ref = "writer")

# ===== CREATE EDITED DATASET ----
data_edited <- data %>%
  group_by(writer_id, proposition_id) %>%
  filter(!(paragraph_type == "model" &
             any(paragraph_type == "edited"))) %>%
  mutate(
    paragraph_type = if_else(paragraph_type == "edited",
                             "model",
                             paragraph_type),
    paragraph_type_ = factor(paragraph_type)
  ) %>%
  ungroup()

data_edited$paragraph_type_ <- relevel(data_edited$paragraph_type_, ref = "writer")

rm(data)

# ===== SIMPLE T-TEST FUNCTION ----

run_test_by_type <- function(df, attribute) {
  
  message("Running test for: ", attribute)
  
  df_sub <- df %>%
    select(paragraph_type_, all_of(attribute)) %>%
    drop_na()
  
  writer_vals <- df_sub %>%
    filter(paragraph_type_ == "writer") %>%
    pull(attribute)
  
  model_vals <- df_sub %>%
    filter(paragraph_type_ == "model") %>%
    pull(attribute)
  
  # Independent t-test
  t_out <- t.test(model_vals, writer_vals)
  
  # Cohen's d
  pooled_sd <- sqrt(
    ((length(model_vals) - 1) * var(model_vals) +
       (length(writer_vals) - 1) * var(writer_vals)) /
      (length(model_vals) + length(writer_vals) - 2)
  )
  
  d <- (mean(model_vals) - mean(writer_vals)) / pooled_sd
  
  tibble(
    mean_writer = mean(writer_vals),
    mean_model = mean(model_vals),
    mean_difference = mean(model_vals) - mean(writer_vals),
    ci_low = t_out$conf.int[1],
    ci_high = t_out$conf.int[2],
    t_statistic = t_out$statistic,
    p_value = t_out$p.value,
    cohens_d = d
  )
}

# ===== ATTRIBUTES ----

rating_attributes <- c(
  "paragraph_formality",
  "paragraph_clarity",
  "paragraph_informativeness",
  "paragraph_originality",
  "paragraph_relevance",
  "writer_knowledge",
  "writer_importance",
  "writer_confidence",
  "writer_stance",
  "writer_stance_polarity",
  "paragraph_hope",
  "paragraph_excitement",
  "paragraph_fear",
  "paragraph_disgust",
  "paragraph_anger",
  "writer_affect_x",
  "writer_affect_y",
  "writer_optimism",
  "writer_community",
  "writer_friendliness",
  "writer_openness"
)

# ===== RUN ALL TESTS (EDITED ONLY) ----

for (attr in rating_attributes) {
  
  results <- run_test_by_type(
    df = data_edited,
    attribute = attr
  )
  
  write_csv(
    results,
    paste0(
      "./results/main_phase_2_distribution/edited/",
      attr,
      "_by_type.csv"
    )
  )
}