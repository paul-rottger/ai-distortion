#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 DISTORTION ANALYSIS: PAIRWISE MODEL COMPARISONS
#
# Tests whether distortion magnitudes differ between AI models.
#
# - Refits the by-model regressions on the preferred subset (beta regressions
#   with crossed reader, writer and proposition random intercepts for scale
#   outcomes; cumulative link mixed models with reader random intercepts for
#   ordinal outcomes), using the same specifications as
#   phase_2_distortion_scale_variables.R and phase_2_distortion_ordinal_variables.R.
# - Scale outcomes: pairwise differences in average marginal effects (AMEs).
# - Ordinal outcomes: pairwise Wald tests on log odds ratios (ratio of ORs).
# - Reports raw and Holm-adjusted (within attribute) p-values.
# - Writes per-attribute CSVs, a summary CSV and a LaTeX block for the SI to
#   results/main_phase_2_distortion/preferred/.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(marginaleffects)
  library(ordinal)
  library(parallel)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility (parallel-safe RNG streams)
RNGkind("L'Ecuyer-CMRG")
set.seed(123)

# Parse command-line flags
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "main_phase_2_distortion", "preferred")
SUMMARY_DIR <- file.path(RESULTS_DIR, "summary")
dir.create(SUMMARY_DIR, recursive = TRUE, showWarnings = FALSE)

# Crossed random intercepts for readers, writers and propositions
RANDOM_EFFECTS <- "(1 | rater_id) + (1 | writer_id) + (1 | proposition_id)"
N_CORES <- min(10, detectCores())
ALPHA <- 0.05

# Model labels and the fixed order of pairwise comparisons
MODEL_LABELS <- c(
  "anthropic/claude-sonnet-4"         = "Claude",
  "openai/chatgpt-4o-latest"          = "ChatGPT",
  "deepseek/deepseek-chat-v3-0324"    = "DeepSeek"
)
MODEL_PAIRS <- list(
  c("anthropic/claude-sonnet-4", "openai/chatgpt-4o-latest"),
  c("anthropic/claude-sonnet-4", "deepseek/deepseek-chat-v3-0324"),
  c("openai/chatgpt-4o-latest", "deepseek/deepseek-chat-v3-0324")
)
pair_label <- function(pair) paste0(MODEL_LABELS[pair[1]], " vs ", MODEL_LABELS[pair[2]])

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

list2env(load_phase2_splits(
  "./data/main_phase_2/annotations.csv",
  "./data/main_phase_1/proposition_responses.csv",
  extra_mutate = function(data) {
    data %>%
      mutate(
        model_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
        across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
      )
  }
), envir = environment())

data_model <- data_preferred
if (demo_mode) {
  message("Running in demo mode on n=1000 samples from the preferred split.")
  data_model <- data_model %>% slice_sample(n = min(1000, nrow(data_model)))
}

# =============================================================================
# ANALYSIS: SCALE VARIABLES (PAIRWISE AME DIFFERENCES)
# =============================================================================

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

pairwise_scale <- function(attribute) {
  print(paste("running pairwise scale comparisons for:", attribute))

  # 0–100 -> proportion, then squeeze to (0,1)
  df <- data_model
  y <- df[[attribute]] / 100
  y <- pmin(pmax(y, 0), 1)
  df$y <- squeeze01(y)

  form <- as.formula(paste0("y ~ model_ + ", RANDOM_EFFECTS))
  model <- glmmTMB(form, data = df, family = beta_family(link = "logit"))

  # AMEs of each model vs writer (one row per model, in factor-level order)
  ame_args <- list(
    model,
    variables = list(model_ = "reference"),
    type = "response",
    re.form = NA
  )
  ame <- do.call(avg_comparisons, ame_args)
  ame_models <- sub(" - writer$", "", ame$contrast)

  # Contrast matrix: one column per model pair (AME_i - AME_j)
  hyp <- sapply(MODEL_PAIRS, function(pair) {
    as.numeric(ame_models == pair[1]) - as.numeric(ame_models == pair[2])
  })
  colnames(hyp) <- sapply(MODEL_PAIRS, pair_label)

  diff <- as_tibble(do.call(avg_comparisons, c(ame_args, list(hypothesis = hyp))))

  tibble(
    attribute = attribute,
    variable_type = "scale",
    pair = colnames(hyp),
    model_a = sapply(MODEL_PAIRS, `[`, 1),
    model_b = sapply(MODEL_PAIRS, `[`, 2),
    estimate = diff$estimate * 100,
    conf_low = diff$conf.low * 100,
    conf_high = diff$conf.high * 100,
    statistic = diff$statistic,
    p = diff$p.value
  )
}

# =============================================================================
# ANALYSIS: ORDINAL VARIABLES (PAIRWISE LOG-OR DIFFERENCES)
# =============================================================================

pairwise_ordinal <- function(attribute) {
  print(paste("running pairwise ordinal comparisons for:", attribute))

  df <- data_model %>%
    filter(!is.na(.data[[attribute]]), !is.na(model_)) %>%
    mutate(rater_id = as.factor(rater_id))

  # Drop "Other" only when fitting writer_education model
  if (attribute == "writer_education") {
    df <- df %>% filter(writer_education != "Other")
  }

  # Cumulative link mixed model (logit link), as in phase_2_distortion_ordinal_variables.R
  form <- as.formula(paste0(attribute, " ~ model_ + (1 | rater_id)"))
  model <- clmm(
    form,
    data = df,
    link = "logit",
    Hess = TRUE,
    nAGQ = 0,
    control = clmm.control(maxIter = 200, gradTol = 1e-4)
  )

  # If the Hessian is not positive definite (only seen on small demo
  # subsamples), fall back to point estimates without CIs.
  b <- coef(model)
  V <- tryCatch(vcov(model), error = function(e) {
    warning(sprintf("OR ratio CIs unavailable for %s: %s", attribute, conditionMessage(e)), call. = FALSE)
    matrix(NA_real_, length(b), length(b), dimnames = list(names(b), names(b)))
  })

  map_dfr(MODEL_PAIRS, function(model_pair) {
    ta <- paste0("model_", model_pair[1])
    tb <- paste0("model_", model_pair[2])
    est <- unname(b[ta] - b[tb])
    se <- sqrt(V[ta, ta] + V[tb, tb] - 2 * V[ta, tb])
    z <- est / se
    tibble(
      attribute = attribute,
      variable_type = "ordinal",
      pair = pair_label(model_pair),
      model_a = model_pair[1],
      model_b = model_pair[2],
      # Ratio of odds ratios (OR_a / OR_b)
      estimate = exp(est),
      conf_low = exp(est - qnorm(0.975) * se),
      conf_high = exp(est + qnorm(0.975) * se),
      statistic = z,
      p = 2 * pnorm(-abs(z))
    )
  })
}

# =============================================================================
# RUN ANALYSES
# =============================================================================

scale_status <- mclapply(rating_attributes, function(a) try(pairwise_scale(a)), mc.cores = N_CORES, mc.set.seed = TRUE)
failed <- vapply(scale_status, inherits, logical(1), what = "try-error")
if (any(failed)) {
  stop("Pairwise scale comparisons failed for: ", paste(rating_attributes[failed], collapse = ", "))
}

ordinal_status <- mclapply(ordinal_vars, function(a) try(pairwise_ordinal(a)), mc.cores = N_CORES, mc.set.seed = TRUE)
failed <- vapply(ordinal_status, inherits, logical(1), what = "try-error")
if (any(failed)) {
  stop("Pairwise ordinal comparisons failed for: ", paste(ordinal_vars[failed], collapse = ", "))
}

results <- bind_rows(scale_status, ordinal_status) %>%
  group_by(attribute) %>%
  mutate(p_holm = p.adjust(p, method = "holm")) %>%
  ungroup()

# Per-attribute result files
for (a in unique(results$attribute)) {
  write_csv(
    results %>% filter(attribute == a),
    file.path(RESULTS_DIR, paste0(a, "_by_model_pairwise.csv"))
  )
}

# Summary: attributes with at least one significant model pair
summary_tbl <- results %>%
  group_by(variable_type, attribute) %>%
  summarise(
    any_sig_raw = any(p < ALPHA, na.rm = TRUE),
    any_sig_holm = any(p_holm < ALPHA, na.rm = TRUE),
    n_sig_pairs_raw = sum(p < ALPHA, na.rm = TRUE),
    n_sig_pairs_holm = sum(p_holm < ALPHA, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(variable_type) %>%
  summarise(
    n_attributes = n(),
    n_attributes_any_sig_raw = sum(any_sig_raw),
    n_attributes_any_sig_holm = sum(any_sig_holm),
    n_pairs = 3 * n(),
    n_pairs_sig_raw = sum(n_sig_pairs_raw),
    n_pairs_sig_holm = sum(n_sig_pairs_holm),
    .groups = "drop"
  )
write_csv(summary_tbl, file.path(SUMMARY_DIR, "phase_2_distortion_by_model_pairwise_summary.csv"))
print(summary_tbl)
