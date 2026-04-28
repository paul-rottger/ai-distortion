#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 1 ANALYSIS: PARAGRAPH PREFERENCE
# 
# Summarizes writer preferences between original and model-edited paragraphs.
#
# - Computes preference and edit rates with bootstrap confidence intervals.
# - Summarizes strict preference reasons for AI vs human selections.
# - Fits mixed-effects logistic regressions for binary preference/editing outcomes.
# - Fits mixed-effects linear regressions for continuous edit outcomes.
# - Writes result tables to results/main_phase_1/ and figures to figures/main_phase_1/.
# 
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
})

source("./analysis/utils_r/demo_paths.R")

# Set random seed for reproducibility
set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_1")
FIGURES_DIR <- get_figures_dir(demo_mode, "main_phase_1")

# =============================================================================
# DATA LOADING
# =============================================================================

data <- read_csv("./data/main_phase_1/proposition_responses.csv",
  show_col_types = FALSE
)
edit_metrics <- read_csv(
  file.path(RESULTS_DIR, "paragraph_edit_ratio_by_writer_proposition.csv"),
  show_col_types = FALSE
) %>%
  dplyr::select(writer_id, proposition_id, edit_ratio, edit_distance)

# =============================================================================
# DATA PROCESSING
# =============================================================================

data <- data %>%
  left_join(edit_metrics, by = c("writer_id", "proposition_id")) %>%
  mutate(
    model_ = as.factor(model_name),
    input_condition_ = as.factor(model_input_condition),
    writer_id = as.factor(writer_id),
    edit_ratio = replace_na(edit_ratio, 0),
    edit_distance = replace_na(edit_distance, 0),
    weak_preference_model = writer_preference != "original",
    strict_preference_model = writer_preference == "edited",
    made_edits = as.integer(made_edits),
    weak_preference_model = as.integer(weak_preference_model),
    strict_preference_model = as.integer(strict_preference_model),
  )

if (demo_mode) {
  data <- data %>%
    slice_sample(n = min(1000, nrow(data)))

  message("Running in demo mode on an n=1000 sample of the phase 1 data.")
}

main_phase_1_group_levels <- c(
  "improve",
  "rewrite",
  "bullets-based",
  "stance-based",
  "anthropic/claude-sonnet-4",
  "deepseek/deepseek-chat-v3-0324",
  "openai/chatgpt-4o-latest",
  "overall"
)

clean_reason_text <- function(reason_text) {
  reason_text %>%
    replace_na("") %>%
    str_remove_all("\\[|\\]|'|\"") %>%
    str_squish()
}

summarize_main_phase_1_groups <- function(df, summarize_group_fn) {
  bind_rows(
    summarize_group_fn(df) %>%
      mutate(group = "overall"),

    df %>%
      group_by(model_) %>%
      group_modify(~ summarize_group_fn(.x)) %>%
      mutate(group = as.character(model_)) %>%
      ungroup() %>%
      dplyr::select(-model_),

    df %>%
      group_by(input_condition_) %>%
      group_modify(~ summarize_group_fn(.x)) %>%
      mutate(group = as.character(input_condition_)) %>%
      ungroup() %>%
      dplyr::select(-input_condition_)
  ) %>%
    mutate(group = factor(group, levels = main_phase_1_group_levels)) %>%
    arrange(group)
}

add_main_phase_1_group_labels <- function(table) {
  group_label_levels <- table %>%
    mutate(group = as.character(group)) %>%
    distinct(group, n) %>%
    mutate(group = factor(group, levels = main_phase_1_group_levels)) %>%
    arrange(group) %>%
    transmute(group_label = paste0(group, " (n = ", n, ")")) %>%
    pull(group_label)

  table %>%
    mutate(
      group = factor(group, levels = main_phase_1_group_levels),
      group_label = factor(
        paste0(as.character(group), " (n = ", n, ")"),
        levels = group_label_levels
      )
    )
}

# =============================================================================
# ANALYSIS: PREFERENCE RATES + BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

bootstrap_preference_summary <- function(data,
                                         pref_var,
                                         n_boot = 1000,
                                         conf = 0.99) {
  alpha <- (1 - conf) / 2

  summarize_group <- function(df) {
    n <- nrow(df)

    boot_means <- replicate(n_boot, mean(df[[pref_var]][sample.int(nrow(df), replace = TRUE)]))

    tibble(
      n = n,
      prop_preferred = mean(df[[pref_var]]),
      n_preferred = sum(df[[pref_var]]),
      ci_low = quantile(boot_means, probs = alpha),
      ci_high = quantile(boot_means, probs = 1 - alpha)
    )
  }

  table <- summarize_main_phase_1_groups(data, summarize_group) %>%
    dplyr::select(group, n, n_preferred, prop_preferred, ci_low, ci_high) %>%
    add_main_phase_1_group_labels()

  plot <- ggplot(table, aes(x = prop_preferred, y = group_label)) +
    geom_point() +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0.2) +
    labs(
      x = paste0(pref_var, " with ", scales::percent(conf, accuracy = 1), " bootstrap CI"),
      y = NULL,
    ) +
    scale_x_continuous(
      labels = scales::label_percent(accuracy = 1),
      limits = c(0, 1),
      expand = c(0, 0)
    )

  list(table = table, plot = plot)
}

summarize_response_table <- function(data,
                                     response_var,
                                     response_levels,
                                     response_labels,
                                     output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  summarize_group <- function(df) {
    response_counts <- table(factor(as.character(df[[response_var]]), levels = response_levels))

    tibble(
      n = nrow(df),
      response = response_labels,
      n_response = as.integer(response_counts),
      prop_response = as.integer(response_counts) / nrow(df),
      cell = paste0(
        as.integer(response_counts),
        " (",
        scales::percent(as.integer(response_counts) / nrow(df), accuracy = 0.1),
        ")"
      )
    )
  }

  table <- summarize_main_phase_1_groups(data, summarize_group) %>%
    mutate(response = factor(response, levels = response_labels)) %>%
    dplyr::select(group, n, response, cell) %>%
    pivot_wider(names_from = response, values_from = cell) %>%
    mutate(group = factor(group, levels = main_phase_1_group_levels)) %>%
    arrange(group)

  print(table)
  write_csv(table %>% mutate(group = as.character(group)), output_path)

  table
}

produce_results <- function(data, var) {
  summary <- bootstrap_preference_summary(data, var)
  print(summary$table)
  dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)
  ggsave(
    file.path(FIGURES_DIR, paste0(var, ".pdf")),
    summary$plot,
    width = 8,
    height = 4,
    dpi = 150
  )
}

preference_summary_table <- summarize_response_table(
  data = data,
  response_var = "writer_preference",
  response_levels = c("original", "equal", "edited"),
  response_labels = c("human", "equal", "ai"),
  output_path = file.path(RESULTS_DIR, "writer_preference_summary_table.csv")
)

made_edits_summary_table <- summarize_response_table(
  data = data,
  response_var = "made_edits",
  response_levels = c("0", "1"),
  response_labels = c("no", "yes"),
  output_path = file.path(RESULTS_DIR, "made_edits_summary_table.csv")
)

for (var in c(
  "made_edits",
  "weak_preference_model",
  "strict_preference_model"
)) {
  produce_results(data, var)
}

# =============================================================================
# ANALYSIS: PREFERENCE REASONS
# =============================================================================

summarize_strict_preference_reasons <- function(df,
                                                output_path = file.path(RESULTS_DIR, "strict_preference_reason_summary.csv"),
                                                overview_path = file.path(RESULTS_DIR, "strict_preference_reason_overview.csv")) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  strict_preference_data <- df %>%
    mutate(
      response_id = row_number(),
      strict_preference = case_when(
        writer_preference == "edited" ~ "ai",
        writer_preference == "original" ~ "human",
        TRUE ~ NA_character_
      ),
      cleaned_reason_text = clean_reason_text(writer_preference_reason),
      cleaned_reason_other_text = clean_reason_text(writer_preference_reason_other)
    ) %>%
    filter(!is.na(strict_preference)) %>%
    mutate(strict_preference = factor(strict_preference, levels = c("ai", "human")))

  overview <- strict_preference_data %>%
    group_by(strict_preference) %>%
    summarize(
      n_strict_preferences = n(),
      .groups = "drop"
    )

  selected_reason_data <- strict_preference_data %>%
    mutate(writer_preference_reason = str_split(cleaned_reason_text, ",\\s*")) %>%
    unnest(writer_preference_reason) %>%
    mutate(writer_preference_reason = str_trim(writer_preference_reason)) %>%
    filter(writer_preference_reason != "") %>%
    distinct(response_id, strict_preference, writer_preference_reason)

  other_reason_data <- strict_preference_data %>%
    filter(cleaned_reason_other_text != "") %>%
    transmute(
      response_id,
      strict_preference,
      writer_preference_reason = "other"
    )

  strict_reason_data <- bind_rows(selected_reason_data, other_reason_data) %>%
    distinct(response_id, strict_preference, writer_preference_reason)

  reason_summary <- strict_reason_data %>%
    count(strict_preference, writer_preference_reason, sort = TRUE, name = "n_responses") %>%
    left_join(overview, by = "strict_preference") %>%
    mutate(
      prop_strict_preferences = n_responses / n_strict_preferences
    ) %>%
    arrange(strict_preference, desc(n_responses), writer_preference_reason)

  print(overview)
  print(reason_summary)
  write_csv(overview, overview_path)
  write_csv(reason_summary, output_path)

  list(overview = overview, reasons = reason_summary)
}

strict_preference_reason_summary <- summarize_strict_preference_reasons(data)

# =============================================================================
# ANALYSIS: LOGISTIC REGRESSIONS FOR BINARY OUTCOMES (PREFERENCE + EDITING)
# =============================================================================

# ===== MIXED-EFFECTS LOGISTIC REGRESSIONS ----
fit_main_mixed_logit <- function(df,
                                 outcome,
                                 random_effects = "(1 | writer_id)") {
  fixed_effects <- c("model_", "input_condition_")

  if (outcome %in% c("weak_preference_model", "strict_preference_model")) {
    fixed_effects <- c(fixed_effects, "made_edits")
  }

  form <- as.formula(
    paste0(outcome, " ~ ", paste(fixed_effects, collapse = " + "), " + ", random_effects)
  )

  model <- glmmTMB(
    form,
    data = df,
    family = binomial(link = "logit")
  )

  broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      outcome = outcome,
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p_value = p.value,
    ) %>%
    dplyr::select(
      outcome,
      term,
      estimate,
      std.error,
      statistic,
      p_value,
      conf.low,
      conf.high,
      odds_ratio,
      or_low,
      or_high
    )
}

run_main_mixed_models <- function(df,
                                  outcomes = c("made_edits", "weak_preference_model", "strict_preference_model"),
                                  output_dir = RESULTS_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  results_list <- map(outcomes, ~ fit_main_mixed_logit(df, .x))
  names(results_list) <- outcomes

  walk2(results_list, outcomes, ~ {
    write_csv(.x, file.path(output_dir, paste0(.y, "_mixed_logit.csv")))
  })

  bind_rows(results_list)
}

mixed_logit_results <- run_main_mixed_models(data)
print(mixed_logit_results)

# =============================================================================
# ANALYSIS: LINEAR REGRESSIONS FOR CONTINUOUS OUTCOMES (EDITING)
# =============================================================================

fit_main_mixed_linear <- function(df,
                                  outcome,
                                  random_effects = "(1 | writer_id)") {
  form <- as.formula(
    paste0(outcome, " ~ model_ + input_condition_ + ", random_effects)
  )

  model <- glmmTMB(
    form,
    data = df,
    family = gaussian(link = "identity")
  )

  broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      outcome = outcome,
      p_value = p.value,
    ) %>%
    dplyr::select(
      outcome,
      term,
      estimate,
      std.error,
      statistic,
      p_value,
      conf.low,
      conf.high
    )
}

run_main_mixed_linear_models <- function(df,
                                         outcomes = c("edit_distance", "edit_ratio"),
                                         output_dir = RESULTS_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  results_list <- map(outcomes, ~ fit_main_mixed_linear(df, .x))
  names(results_list) <- outcomes

  walk2(results_list, outcomes, ~ {
    write_csv(.x, file.path(output_dir, paste0(.y, "_mixed_linear.csv")))
  })

  bind_rows(results_list)
}

mixed_linear_results <- run_main_mixed_linear_models(data)
print(mixed_linear_results)
