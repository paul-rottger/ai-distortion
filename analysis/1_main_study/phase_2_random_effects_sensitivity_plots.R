#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: RANDOM-EFFECTS STRUCTURE PLOTS
#
# Summarises the fits from phase_2_random_effects_sensitivity.R.
#
# - Writes a convergence/runtime table (one row per fit).
# - Writes a variance-components table with latent-scale variance shares.
# - Plots writer-vs-model estimates across random-effects specs, and the
#   share of latent variance attributable to readers, writers and propositions.
#
# Usage: Rscript phase_2_random_effects_sensitivity_plots.R [demo] [split=preferred]
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

source("./analysis/utils_r/demo_paths.R")

args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
hit <- grep("^split=", args, value = TRUE)
DATA_SPLIT <- if (length(hit) == 0) "preferred" else sub("^split=", "", hit[1])

RESULTS_DIR <- get_results_input_dir(demo_mode, "main_phase_2_random_effects", DATA_SPLIT)
OUT_RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_random_effects", DATA_SPLIT)
FIGURES_DIR <- get_figures_dir(demo_mode, "main_phase_2_random_effects", DATA_SPLIT)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

SPECS <- c("R", "R+P", "R+W", "R+W+P")
SPEC_COLOURS <- c("R" = "#0070C0", "R+P" = "#70AD47", "R+W" = "#ED7D31", "R+W+P" = "#7030A0")

# Display order (top to bottom) for scale attributes, as in phase_2_distortion_plots.py
SCALE_ATTRIBUTES_GROUPED <- c(
  "paragraph_formality", "paragraph_informativeness", "paragraph_originality", "paragraph_clarity",
  "paragraph_relevance", "writer_knowledge", "writer_importance", "writer_confidence",
  "writer_stance_polarity", "writer_openness", "paragraph_hope", "paragraph_excitement",
  "paragraph_fear", "paragraph_disgust", "paragraph_anger", "writer_affect_x", "writer_affect_y",
  "writer_optimism", "writer_community", "writer_friendliness"
)

fits <- read_csv(file.path(RESULTS_DIR, "fits.csv"), show_col_types = FALSE, guess_max = 10000) %>%
  mutate(
    spec = factor(spec, levels = SPECS),
    label = if_else(is.na(target_level), attribute, paste0(attribute, ": ", target_level))
  )

# =============================================================================
# CONVERGENCE / RUNTIME TABLE
# =============================================================================

convergence <- fits %>%
  select(family, attribute, target_level, spec, status, seconds, conv_code, pdhess, max_grad, singular, any_of(c("error", "warnings"))) %>%
  arrange(family, attribute, target_level, spec)

write_csv(convergence, file.path(OUT_RESULTS_DIR, "convergence_summary.csv"))

print(convergence %>% count(family, spec, status) %>% pivot_wider(names_from = status, values_from = n, values_fill = 0))

# =============================================================================
# VARIANCE COMPONENTS (latent logit scale)
# =============================================================================

# Residual variance on the latent logit scale: pi^2/3 for logistic models; for
# beta regression, Var(logit Y) with Y ~ Beta(mu*phi, (1-mu)*phi) evaluated at
# the writer-paragraph mean.
variance_components <- fits %>%
  filter(status %in% c("ok", "warning")) %>%
  distinct(family, attribute, target_level, label, spec, sd_rater, sd_writer, sd_proposition, intercept, phi) %>%
  mutate(
    mu = plogis(intercept),
    var_residual = if_else(
      family == "beta",
      trigamma(mu * phi) + trigamma((1 - mu) * phi),
      pi^2 / 3
    ),
    var_rater = coalesce(sd_rater^2, 0),
    var_writer = coalesce(sd_writer^2, 0),
    var_proposition = coalesce(sd_proposition^2, 0),
    var_total = var_rater + var_writer + var_proposition + var_residual,
    share_rater = var_rater / var_total,
    share_writer = var_writer / var_total,
    share_proposition = var_proposition / var_total,
    share_residual = var_residual / var_total
  ) %>%
  select(-mu)

write_csv(variance_components, file.path(OUT_RESULTS_DIR, "variance_components.csv"))

shares_plot_data <- variance_components %>%
  filter(spec == "R+W+P") %>%
  select(family, label, starts_with("share_")) %>%
  pivot_longer(starts_with("share_"), names_to = "component", values_to = "share") %>%
  mutate(
    component = factor(
      sub("share_", "", component),
      levels = c("residual", "proposition", "writer", "rater"),
      labels = c("Residual", "Proposition", "Writer", "Reader")
    ),
    family = factor(family, levels = c("beta", "ova"), labels = c("Scale (beta)", "Nominal (one-vs-all logit)"))
  )

if (nrow(shares_plot_data) > 0) {
  shares_figure <- ggplot(shares_plot_data, aes(x = share, y = label, fill = component)) +
    geom_col(width = 0.75) +
    facet_grid(family ~ ., scales = "free_y", space = "free_y") +
    scale_x_continuous(labels = scales::percent, expand = c(0, 0)) +
    scale_fill_manual(
      values = c(Reader = "#0070C0", Writer = "#ED7D31", Proposition = "#70AD47", Residual = "grey85"),
      breaks = c("Reader", "Writer", "Proposition", "Residual")
    ) +
    labs(x = "Share of latent (logit-scale) variance, R+W+P model", y = NULL, fill = NULL) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", panel.grid.major.y = element_blank(), strip.text.y = element_text(angle = 0, hjust = 0))

  ggsave(file.path(FIGURES_DIR, "variance_shares.pdf"), shares_figure, width = 8, height = 11)
}

# =============================================================================
# ESTIMATES ACROSS RANDOM-EFFECTS SPECIFICATIONS
# =============================================================================

estimates <- fits %>%
  filter(status %in% c("ok", "warning")) %>%
  mutate(
    value = if_else(family == "beta", ame, exp(estimate)),
    low = if_else(family == "beta", ame_low, exp(conf_low)),
    high = if_else(family == "beta", ame_high, exp(conf_high))
  )

plot_estimates <- function(df, xlab, ref, log_x = FALSE) {
  p <- ggplot(df, aes(x = value, xmin = low, xmax = high, y = label, colour = spec)) +
    geom_vline(xintercept = ref, linewidth = 0.3, colour = "grey50") +
    geom_pointrange(position = position_dodge(width = 0.7), size = 0.15, linewidth = 0.4) +
    scale_colour_manual(values = SPEC_COLOURS, drop = FALSE) +
    scale_y_discrete(limits = rev) +
    labs(x = xlab, y = NULL, colour = "Random effects") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", panel.grid.major.y = element_blank())
  if (log_x) p <- p + scale_x_log10()
  p
}

beta_est <- estimates %>%
  filter(family == "beta") %>%
  mutate(label = factor(label, levels = SCALE_ATTRIBUTES_GROUPED))
if (nrow(beta_est) > 0) {
  ggsave(
    file.path(FIGURES_DIR, "estimates_scale_ame.pdf"),
    plot_estimates(beta_est, "AME of AI-assisted vs human writing (0-100 scale)", 0),
    width = 7, height = 8
  )
}

or_est <- estimates %>% filter(family == "ova")
if (nrow(or_est) > 0) {
  ggsave(
    file.path(FIGURES_DIR, "estimates_nominal_or.pdf"),
    plot_estimates(or_est, "Odds ratio of AI-assisted vs human writing", 1, log_x = TRUE),
    width = 7, height = 9
  )
}

# Compact comparison against the reader-only specification
comparison <- estimates %>%
  select(family, attribute, target_level, spec, estimate, std_error, se_ratio_vs_r, estimate_diff_vs_r, delta_aic_vs_r, lrt_p_vs_r) %>%
  arrange(family, attribute, target_level, spec)

write_csv(comparison, file.path(OUT_RESULTS_DIR, "estimate_comparison.csv"))

message("Wrote summary tables to ", OUT_RESULTS_DIR, " and figures to ", FIGURES_DIR)
