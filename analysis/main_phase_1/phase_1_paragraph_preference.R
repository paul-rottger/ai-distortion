# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
  library(glmmTMB)
  library(broom.mixed)
})

# ===== PLOTTING DEFAULTS ----
font_add(family = "CMU Serif", regular = "~/Library/Fonts/cmunrm.ttf")
showtext_auto()
theme_set(theme_minimal(base_family = "CMU Serif", base_size = 14))

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/main_phase_1/proposition_responses.csv",
  show_col_types = FALSE
)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    model_ = as.factor(model_name),
    input_condition_ = as.factor(model_input_condition),
    writer_id = as.factor(writer_id),
    weak_preference_model = writer_preference != "original",
    strict_preference_model = writer_preference == "edited",
    made_edits = as.integer(made_edits),
    weak_preference_model = as.integer(weak_preference_model),
    strict_preference_model = as.integer(strict_preference_model),
  )

# ===== PREFERENCE RATES + CIs ----
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

  table <- bind_rows(
    # overall
    data %>%
      summarize_group() %>%
      mutate(group = "overall"),

    # by model
    data %>%
      group_by(model_) %>%
      group_modify(~ summarize_group(.x)) %>%
      mutate(group = as.character(model_)) %>%
      ungroup(),

    # by input condition
    data %>%
      group_by(input_condition_) %>%
      group_modify(~ summarize_group(.x)) %>%
      mutate(group = as.character(input_condition_)) %>%
      ungroup()
  ) %>%
    dplyr::select(group, n, n_preferred, prop_preferred, ci_low, ci_high)

  table <- table %>%
    mutate(
      group = factor(
        group,
        levels = c(
          "improve",
          "rewrite",
          "bullets-based",
          "stance-based",
          "anthropic/claude-sonnet-4",
          "deepseek/deepseek-chat-v3-0324",
          "openai/chatgpt-4o-latest",
          "overall"
        )
      ),
      group_label = factor(paste0(group, " (n = ", n, ")"), levels = paste0(levels(group), " (n = ", n[match(levels(group), group)], ")"))
    )

  plot <- ggplot(table, aes(x = prop_preferred, y = group_label)) +
    geom_point() +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0.2) +
    labs(
      x = paste0("% ", pref_var, " with 95% bootstrap CI"),
      y = NULL,
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0))

  list(table = table, plot = plot)
}

produce_results <- function(data, var) {
  summary <- bootstrap_preference_summary(data, var)
  print(summary$table)
  print(summary$plot)
  ggsave(
    paste0("./figures/main_phase_1/", var, ".pdf"),
    summary$plot,
    width = 8,
    height = 4,
    dpi = 150
  )
}

for (var in c(
  "made_edits",
  "weak_preference_model",
  "strict_preference_model"
)) {
  produce_results(data, var)
}

# ===== STRICT AI PREFERENCE REASONS ----
summarize_strict_preference_reasons <- function(df,
                                                output_path = "./results/main_phase_1/strict_preference_reason_summary.csv") {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  strict_reason_data <- df %>%
    filter(strict_preference_model == 1) %>%
    mutate(
      response_id = row_number(),
      writer_preference_reason = replace_na(writer_preference_reason, ""),
      writer_preference_reason = str_remove_all(writer_preference_reason, "\\[|\\]|'|\""),
      writer_preference_reason = str_split(writer_preference_reason, ",\\s*")
    ) %>%
    unnest(writer_preference_reason) %>%
    mutate(writer_preference_reason = str_trim(writer_preference_reason)) %>%
    filter(writer_preference_reason != "") %>%
    distinct(response_id, writer_preference_reason)

  n_strict_preferences <- df %>%
    filter(strict_preference_model == 1) %>%
    nrow()

  reason_summary <- strict_reason_data %>%
    count(writer_preference_reason, sort = TRUE, name = "n_responses") %>%
    mutate(
      n_strict_preferences = n_strict_preferences,
      prop_strict_preferences = n_responses / n_strict_preferences
    )

  print(reason_summary)
  write_csv(reason_summary, output_path)

  reason_summary
}

strict_preference_reason_summary <- summarize_strict_preference_reasons(data)

# ===== MIXED-EFFECTS LOGISTIC REGRESSIONS ----
fit_main_mixed_logit <- function(df,
                                 outcome,
                                 random_effects = "(1 | writer_id)") {
  form <- as.formula(paste0(outcome, " ~ model_ + input_condition_ + ", random_effects))

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
                                  output_dir = "./results/main_phase_1") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  results_list <- map(outcomes, ~ fit_main_mixed_logit(df, .x))
  names(results_list) <- outcomes

  walk2(results_list, outcomes, ~ {
    write_csv(.x, file.path(output_dir, paste0(.y, "_mixed_logit_main_phase_1.csv")))
  })

  bind_rows(results_list)
}

mixed_logit_results <- run_main_mixed_models(data)
print(mixed_logit_results)
