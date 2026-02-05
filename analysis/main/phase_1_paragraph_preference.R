# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(showtext)
  library(systemfonts)
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
                 show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
  mutate(
    model_ = as.factor(model_name),
    input_condition_ = as.factor(model_input_condition),
    weak_preference_model = writer_preference != "original",
    strict_preference_model = writer_preference == "edited",
  )

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
  
  table <- bind_rows(
    # overall
    data %>%
      summarize_group() %>%
      mutate(group = "overall"),
    
    # by model
    data %>%
      group_by(model_) %>%
      group_modify( ~ summarize_group(.x)) %>%
      mutate(group = as.character(model_)) %>%
      ungroup(),
    
    # by input condition
    data %>%
      group_by(input_condition_) %>%
      group_modify( ~ summarize_group(.x)) %>%
      mutate(group = as.character(input_condition_)) %>%
      ungroup()
  ) %>%
    select(group, n, prop_preferred, ci_low, ci_high)
  
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
    labs(x = paste0("% ", pref_var, " with 95% bootstrap CI"),
         y = NULL,) +
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

for (var in c("made_edits",
              "weak_preference_model",
              "strict_preference_model")) {
  produce_results(data, var)
}