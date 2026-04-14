# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
})

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/followup_mitigation_phase_1/proposition_responses.csv",
                 show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    model_ = as.factor(model_name),
    input_condition_ = as.factor(model_input_condition),
    mitigation_condition_ = as.factor(model_mitigation_condition),
    writer_id = as.factor(writer_id),
    weak_preference_model = writer_preference != "original",
    strict_preference_model = writer_preference == "edited",
    made_edits = as.integer(made_edits),
    weak_preference_model = as.integer(weak_preference_model),
    strict_preference_model = as.integer(strict_preference_model),
  )

data$mitigation_condition_ <- relevel(data$mitigation_condition_, ref = "none")
data$model_ <- relevel(data$model_, ref = "anthropic/claude-sonnet-4")

mitigation_group_levels <- c(
  "reranking | deepseek/deepseek-chat-v3-0324",
  "prompting | deepseek/deepseek-chat-v3-0324",
  "none | deepseek/deepseek-chat-v3-0324",
  "reranking | openai/chatgpt-4o-latest",
  "prompting | openai/chatgpt-4o-latest",
  "none | openai/chatgpt-4o-latest",
  "reranking | anthropic/claude-sonnet-4",
  "prompting | anthropic/claude-sonnet-4",
  "none | anthropic/claude-sonnet-4",
  "reranking",
  "prompting",
  "none"
)

summarize_mitigation_groups <- function(df, summarize_group_fn) {
  bind_rows(
    df %>%
      group_by(mitigation_condition_) %>%
      group_modify(~ summarize_group_fn(.x)) %>%
      mutate(group = as.character(mitigation_condition_)) %>%
      ungroup() %>%
      dplyr::select(-mitigation_condition_),

    df %>%
      group_by(mitigation_condition_, model_) %>%
      group_modify(~ summarize_group_fn(.x)) %>%
      mutate(group = paste(as.character(mitigation_condition_), as.character(model_), sep = " | ")) %>%
      ungroup() %>%
      dplyr::select(-mitigation_condition_, -model_)
  ) %>%
    mutate(group = factor(group, levels = mitigation_group_levels)) %>%
    arrange(group)
}

add_mitigation_group_labels <- function(table) {
  group_label_levels <- table %>%
    mutate(group = as.character(group)) %>%
    distinct(group, n) %>%
    mutate(group = factor(group, levels = mitigation_group_levels)) %>%
    arrange(group) %>%
    transmute(group_label = paste0(group, " (n = ", n, ")")) %>%
    pull(group_label)

  table %>%
    mutate(
      group = factor(group, levels = mitigation_group_levels),
      group_label = factor(
        paste0(as.character(group), " (n = ", n, ")"),
        levels = group_label_levels
      )
    )
}

# ===== PREFERENCE RATES + CIs ----
bootstrap_preference_summary <- function(data,
                                         pref_var,
                                         n_boot = 1000,
                                         conf = 0.95) {
  alpha <- (1 - conf) / 2
  
  summarize_group <- function(df) {
    n <- nrow(df)
    
    boot_means <- replicate(n_boot, mean(df[[pref_var]][sample.int(nrow(df), replace = TRUE)]))
    
    tibble(
      n = n,
      prop_preferred = mean(df[[pref_var]]),
      ci_low = quantile(boot_means, probs = alpha),
      ci_high = quantile(boot_means, probs = 1 - alpha)
    )
  }
  
  table <- summarize_mitigation_groups(data, summarize_group) %>%
    select(group, n, prop_preferred, ci_low, ci_high) %>%
    add_mitigation_group_labels()
  
  plot <- ggplot(table, aes(x = prop_preferred, y = group_label)) +
    geom_point() +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = 0.2) +
    labs(x = paste0("% ", pref_var, " with 95% bootstrap CI"),
         y = NULL,
    ) +
    geom_hline(yintercept = c(3.5, 6.5, 9.5), linetype = "dotted")+
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.1),
      expand = c(0, 0),
      labels = scales::label_percent(accuracy = 1)
    )
  
  list(table = table, plot = plot)
}


produce_results <- function(data, var) {
  summary <- bootstrap_preference_summary(data, var)
  print(summary$table)
  plot_height <- max(4, 0.45 * nrow(summary$table) + 1)
  ggsave(
    paste0("./figures/followup_mitigation_phase_1/", var, ".pdf"),
    summary$plot,
    width = 8,
    height = plot_height,
    dpi = 150
  )
}

for (var in c("made_edits",
              "weak_preference_model",
              "strict_preference_model")) {
  produce_results(data, var)
}

# ===== MIXED-EFFECTS LOGISTIC REGRESSIONS ----
fit_mitigation_mixed_logit <- function(df,
                                       outcome,
                                       random_effects = "(1 | writer_id)") {
  fixed_effects <- c("mitigation_condition_", "model_")

  if (n_distinct(df$input_condition_) > 1) {
    fixed_effects <- c(fixed_effects, "input_condition_")
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
    select(
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

run_mitigation_mixed_models <- function(df,
                                        outcomes = c("made_edits", "weak_preference_model", "strict_preference_model"),
                                        output_dir = "./results/followup_mitigation_phase_1") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  results_list <- map(outcomes, ~ fit_mitigation_mixed_logit(df, .x))
  names(results_list) <- outcomes

  walk2(results_list, outcomes, ~ {
    write_csv(.x, file.path(output_dir, paste0(.y, "_mixed_logit_mitigation_condition.csv")))
  })

  bind_rows(results_list)
}

mixed_logit_results <- run_mitigation_mixed_models(data)
print(mixed_logit_results)