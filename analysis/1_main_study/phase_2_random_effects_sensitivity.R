#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: RANDOM-EFFECTS STRUCTURE
#
# Compares random-effects structures for the overall writer-vs-model distortion
# models (paragraph_type_ predictor) and decomposes variance (supplement).
#
# - Fits beta regressions (scale attributes) and one-vs-all logistic
#   regressions (nominal attribute categories) under four random-intercept
#   specifications: reader only (R), R + proposition (R+P), R + writer (R+W),
#   and R + writer + proposition (R+W+P, pre-registered and used in the main
#   analyses).
# - Records runtime, convergence diagnostics, random-effect SDs, fit
#   statistics, and paragraph-type estimates for each fit.
# - Writes fits.csv to results/main_phase_2_random_effects/<split>/.
#
# Usage: Rscript phase_2_random_effects_sensitivity.R [demo] [split=preferred] [workers=10]
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(marginaleffects)
  library(parallel)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)

get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (length(hit) == 0) default else sub(paste0("^", name, "="), "", hit[1])
}

DATA_SPLIT <- get_arg("split", "preferred")
N_WORKERS  <- as.integer(get_arg("workers", min(10, detectCores())))

RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_random_effects", DATA_SPLIT)
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Random-effects ladder
RE_SPECS <- list(
  "R"     = c("rater_id"),
  "R+P"   = c("rater_id", "proposition_id"),
  "R+W"   = c("rater_id", "writer_id"),
  "R+W+P" = c("rater_id", "writer_id", "proposition_id")
)

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv"
), envir = environment())

split_data <- get_split_data(DATA_SPLIT)

if (demo_mode) {
  message("Running in demo mode on n=1000 samples from the data split.")
  split_data <- split_data %>% slice_sample(n = min(1000, nrow(split_data)))
}

split_data <- split_data %>%
  mutate(paragraph_type_ = relevel(factor(paragraph_type_), ref = "writer"))

# =============================================================================
# HELPERS
# =============================================================================

# Squeeze 0..1 for beta regression (as in phase_2_distortion_scale_variables.R)
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

re_formula <- function(re_terms) {
  paste0("(1 | ", re_terms, ")", collapse = " + ")
}

# Run expr while collecting (and muffling) warnings
with_warnings <- function(expr) {
  warns <- character(0)
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warns))
}

glmmtmb_diagnostics <- function(model) {
  sds <- sapply(VarCorr(model)$cond, function(x) attr(x, "stddev")[[1]])
  tibble(
    conv_code = model$fit$convergence,
    pdhess = model$sdr$pdHess,
    max_grad = max(abs(model$sdr$gradient.fixed)),
    singular = any(sds < 1e-4),
    n_obs = nobs(model),
    loglik = as.numeric(logLik(model)),
    df = attr(logLik(model), "df"),
    aic = AIC(model),
    bic = BIC(model),
    intercept = unname(fixef(model)$cond["(Intercept)"]),
    sd_rater = unname(sds["rater_id"]),
    sd_writer = unname(sds["writer_id"]),
    sd_proposition = unname(sds["proposition_id"])
  )
}

glmmtmb_term <- function(model, term = "paragraph_type_model") {
  co <- summary(model)$coefficients$cond
  est <- co[term, "Estimate"]
  se <- co[term, "Std. Error"]
  tibble(
    estimate = est,
    std_error = se,
    conf_low = est - 1.96 * se,
    conf_high = est + 1.96 * se,
    p = co[term, "Pr(>|z|)"]
  )
}

# =============================================================================
# FIT FUNCTIONS
# =============================================================================

fit_beta <- function(df, outcome, re_terms) {
  df <- df %>% filter(!is.na(.data[[outcome]]))
  y <- df[[outcome]] / 100
  y <- pmin(pmax(y, 0), 1)
  df$y <- squeeze01(y)

  form <- as.formula(paste0("y ~ paragraph_type_ + ", re_formula(re_terms)))
  fit <- with_warnings(glmmTMB(form, data = df, family = beta_family(link = "logit")))
  model <- fit$value

  ame_res <- avg_comparisons(
    model,
    variables = list(paragraph_type_ = "reference"),
    type = "response",
    re.form = NA
  )

  glmmtmb_diagnostics(model) %>%
    bind_cols(glmmtmb_term(model)) %>%
    mutate(
      phi = sigma(model),
      ame = ame_res$estimate[1] * 100,
      ame_low = ame_res$conf.low[1] * 100,
      ame_high = ame_res$conf.high[1] * 100,
      warnings = paste(fit$warnings, collapse = " | ")
    )
}

fit_ova <- function(df, outcome, target_level, re_terms) {
  df <- df %>%
    filter(!is.na(.data[[outcome]])) %>%
    mutate(target_flag = as.integer(.data[[outcome]] == target_level))

  form <- as.formula(paste0("target_flag ~ paragraph_type_ + ", re_formula(re_terms)))
  fit <- with_warnings(glmmTMB(form, data = df, family = binomial(link = "logit")))
  model <- fit$value

  glmmtmb_diagnostics(model) %>%
    bind_cols(glmmtmb_term(model)) %>%
    mutate(warnings = paste(fit$warnings, collapse = " | "))
}

# =============================================================================
# RUN FITS
# =============================================================================

nominal_levels <- map(set_names(nominal_vars), function(v) {
  sort(unique(na.omit(split_data[[v]])))
})

tasks <- bind_rows(
  expand_grid(family = "beta", attribute = rating_attributes, target_level = NA_character_),
  map_dfr(nominal_vars, function(v) tibble(family = "ova", attribute = v, target_level = nominal_levels[[v]]))
) %>%
  expand_grid(spec = names(RE_SPECS)) %>%
  # Start the slowest (fullest) specifications first
  arrange(desc(match(spec, names(RE_SPECS))))

message(sprintf("Fitting %d models (split=%s, workers=%d)", nrow(tasks), DATA_SPLIT, N_WORKERS))

run_task <- function(i) {
  task <- tasks[i, ]
  re_terms <- RE_SPECS[[task$spec]]
  t0 <- Sys.time()
  result <- tryCatch(
    switch(task$family,
      beta = fit_beta(split_data, task$attribute, re_terms),
      ova  = fit_ova(split_data, task$attribute, task$target_level, re_terms)
    ),
    error = function(e) tibble(error = conditionMessage(e))
  )
  seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  message(sprintf("[done] %s %s %s %s (%.0fs)", task$family, task$attribute, coalesce(task$target_level, ""), task$spec, seconds))

  task %>%
    mutate(split = DATA_SPLIT, seconds = seconds, .before = 1) %>%
    bind_cols(result) %>%
    mutate(status = case_when(
      "error" %in% names(result) ~ "error",
      nzchar(warnings) ~ "warning",
      TRUE ~ "ok"
    ), .after = spec)
}

fits <- bind_rows(mclapply(seq_len(nrow(tasks)), run_task, mc.cores = N_WORKERS, mc.preschedule = FALSE))

# =============================================================================
# COMPARISONS AGAINST THE READER-ONLY SPECIFICATION
# =============================================================================

# Likelihood-ratio tests against the reader-only model (naive chi-square p,
# conservative because variance parameters are tested on the boundary)
fits <- fits %>%
  group_by(family, attribute, target_level) %>%
  mutate(
    delta_aic_vs_r = aic - aic[spec == "R"][1],
    lrt_vs_r = 2 * (loglik - loglik[spec == "R"][1]),
    lrt_p_vs_r = if_else(spec == "R", NA_real_, pchisq(lrt_vs_r, df - df[spec == "R"][1], lower.tail = FALSE)),
    se_ratio_vs_r = std_error / std_error[spec == "R"][1],
    estimate_diff_vs_r = estimate - estimate[spec == "R"][1]
  ) %>%
  ungroup() %>%
  mutate(spec = factor(spec, levels = names(RE_SPECS))) %>%
  arrange(family, attribute, target_level, spec)

write_csv(fits, file.path(RESULTS_DIR, "fits.csv"))
message(sprintf("Wrote %d rows to %s", nrow(fits), file.path(RESULTS_DIR, "fits.csv")))
