#!/usr/bin/env Rscript

# =============================================================================
# SENSITIVITY ANALYSIS - MINIMUM DETECTABLE EFFECTS (ALL STUDIES)
#
# Design-based sensitivity analysis on the realised data:
# - Part A: for headline contrasts in Studies 1-5, recover the standard error
#   from reported 95% CIs and compute the minimum detectable effect at 80%
#   power (two-sided alpha = .05): MDE80 = (z_.975 + z_.80) * SE ~= 2.80 * SE.
#   For Studies 4-5, also compute power for the effects ASSUMED in the
#   power analyses described in the Study 4 and Study 5 preregistrations, given the
#   realised SE. This is not post-hoc "observed power": the effect is fixed in
#   advance, only the precision is realised.
# - Part B: refit the Study 4 and Study 5 primary models and compare realised
#   variance components to those assumed in the power analyses.
#
# Reads existing results; run after the study-specific analysis scripts.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
})

source("./analysis/utils_r/demo_paths.R")

demo_mode <- parse_demo_mode()

RESULTS_DIR <- get_results_dir(demo_mode, "sensitivity_analysis")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

ALPHA <- 0.05
TARGET_POWER <- 0.80
Z_ALPHA <- qnorm(1 - ALPHA / 2)
MDE_MULTIPLIER <- Z_ALPHA + qnorm(TARGET_POWER)

# =============================================================================
# HELPERS
# =============================================================================

input_path <- function(...) {
  parts <- c(...)
  file.path(do.call(get_results_input_dir, c(list(demo_mode), as.list(head(parts, -1)))), tail(parts, 1))
}

se_from_ci <- function(low, high) (high - low) / (2 * Z_ALPHA)

power_at <- function(effect, se) {
  ncp <- abs(effect) / se
  pnorm(ncp - Z_ALPHA) + pnorm(-ncp - Z_ALPHA)
}

logit <- function(p) log(p / (1 - p))

make_row <- function(study, contrast, scale, estimate, se, assumed = NA_real_) {
  tibble(
    study = study,
    contrast = contrast,
    scale = scale,
    estimate = estimate,
    se = se,
    mde80 = MDE_MULTIPLIER * se,
    assumed_effect = assumed,
    power_at_assumed = if_else(is.na(assumed), NA_real_, power_at(assumed, se))
  )
}

# =============================================================================
# ASSUMPTIONS FROM PRE-REGISTERED POWER ANALYSES
# (as described in the Study 4 and Study 5 preregistrations)
# =============================================================================

# Study 4: cell means in pence (out of 20p).
s4_human_low <- 7.5; s4_ai_low <- 8.0; s4_human_high <- 7.0; s4_ai_high <- 8.5
s4_assumed_main_pence <- mean(c(s4_ai_low, s4_ai_high)) - mean(c(s4_human_low, s4_human_high))
s4_assumed_int_logit <- (logit(s4_ai_high / 20) - logit(s4_human_high / 20)) -
  (logit(s4_ai_low / 20) - logit(s4_human_low / 20))
s4_assumed_sd_reader <- 0.60
s4_assumed_phi <- 3

# Study 5: cell means in attitude points.
s5_human_low <- 2.5; s5_ai_low <- 3.0; s5_human_high <- 1.0; s5_ai_high <- 3.0
s5_assumed_main <- mean(c(s5_ai_low, s5_ai_high)) - mean(c(s5_human_low, s5_human_high))
s5_assumed_int <- (s5_ai_high - s5_human_high) - (s5_ai_low - s5_human_low)
s5_assumed_sd_reader <- 10
s5_assumed_sd_prop <- 6
s5_assumed_sd_resid <- sqrt(23^2 - 6^2)

# =============================================================================
# PART A: MDEs FROM REPORTED CONFIDENCE INTERVALS
# =============================================================================

rows <- list()

# --- Study 1: AI vs human AMEs on 0-100 rating attributes (unedited subset) ---
s1_dir <- get_results_input_dir(demo_mode, "main_phase_2_distortion", "unedited")
s1_files <- list.files(s1_dir, pattern = "_by_type\\.csv$", full.names = TRUE)

s1_ames <- map_dfr(s1_files, function(f) {
  df <- read_csv(f, show_col_types = FALSE)
  if (!"ame" %in% names(df)) return(NULL)
  df %>%
    filter(term == "paragraph_type_model") %>%
    transmute(
      attribute = str_remove(basename(f), "_by_type\\.csv$"),
      estimate = ame,
      se = se_from_ci(ame_low, ame_high)
    )
})

rows[["s1_attr"]] <- pmap_dfr(s1_ames, function(attribute, estimate, se) {
  make_row("Study 1", paste0("AI vs human: ", attribute), "points (0-100)", estimate, se)
})

# --- Study 1: writer preference (between-condition contrasts, log-odds) ---
s1_pref <- read_csv(input_path("main_phase_1", "strict_preference_model_mixed_logit.csv"),
                    show_col_types = FALSE)
rows[["s1_pref"]] <- s1_pref %>%
  filter(str_detect(term, "^model_|^input_condition_")) %>%
  pmap_dfr(function(term, estimate, std.error, ...) {
    make_row("Study 1", paste0("Writer preference: ", term), "log-odds", estimate, std.error)
  })

# --- Study 2: disclaimer effect on writer preference (log-odds) ---
s2_pref <- read_csv(
  input_path("followup_disclaimer_phase_1", "strict_preference_model_mixed_logit_disclaimer_condition.csv"),
  show_col_types = FALSE
)
rows[["s2_pref"]] <- s2_pref %>%
  filter(str_detect(term, "^disclaimer_condition_")) %>%
  pmap_dfr(function(term, estimate, std.error, ...) {
    make_row("Study 2", paste0("Writer preference: ", term), "log-odds", estimate, std.error)
  })

# --- Study 3: polarising distortion by mitigation arm, and arm contrasts ---
s3_pol <- read_csv(
  input_path("followup_mitigation_phase_2_distortion", "unedited", "writer_stance_polarity_by_mitigation.csv"),
  show_col_types = FALSE
) %>%
  mutate(se = se_from_ci(ame_low, ame_high), arm = str_remove(term, "^mitigation_condition_"))

rows[["s3_arm"]] <- pmap_dfr(select(s3_pol, arm, ame, se), function(arm, ame, se) {
  make_row("Study 3", paste0("AI vs human stance polarity: ", arm), "points (0-100)", ame, se)
})

# Arm-vs-no-intervention contrasts, approximating SE by treating arms as independent.
s3_none <- s3_pol %>% filter(arm == "none")
rows[["s3_contrast"]] <- s3_pol %>%
  filter(arm != "none") %>%
  pmap_dfr(function(arm, ame, se, ...) {
    make_row("Study 3", paste0("Polarity AME, ", arm, " vs none (approx.)"), "points (0-100)",
             ame - s3_none$ame, sqrt(se^2 + s3_none$se^2))
  })

s3_pref <- read_csv(
  input_path("followup_mitigation_phase_1", "strict_preference_model_mixed_logit_mitigation_condition.csv"),
  show_col_types = FALSE
)
rows[["s3_pref"]] <- s3_pref %>%
  filter(str_detect(term, "^mitigation_condition_")) %>%
  pmap_dfr(function(term, estimate, std.error, ...) {
    make_row("Study 3", paste0("Writer preference: ", term), "log-odds", estimate, std.error)
  })

# --- Study 4: trust RQ1 (AME, pence) and RQ2 interaction (log-odds) ---
s4_rq1 <- read_csv(input_path("followup_trust", "rq1_ame.csv"), show_col_types = FALSE) %>%
  filter(model == "rq1_primary", term == "paragraph_type_ai_vs_human")
rows[["s4_rq1"]] <- make_row("Study 4", "RQ1: AI vs human trust allocation", "pence (0-20)",
                             s4_rq1$ame_pence, se_from_ci(s4_rq1$ame_low_pence, s4_rq1$ame_high_pence),
                             s4_assumed_main_pence)

s4_rq2 <- read_csv(input_path("followup_trust", "rq2_fixed_effects.csv"), show_col_types = FALSE) %>%
  filter(model == "rq2_interaction_bin", term == "paragraph_typeai:distortion_binhigh_distortion")
rows[["s4_rq2"]] <- make_row("Study 4", "RQ2: AI x high distortion interaction", "log-odds",
                             s4_rq2$estimate, se_from_ci(s4_rq2$conf.low, s4_rq2$conf.high),
                             s4_assumed_int_logit)

# --- Study 5: persuasion RQ1 and RQ2 dose-response (attitude points) ---
s5_rq1 <- read_csv(input_path("followup_persuasion", "rq1_ame.csv"), show_col_types = FALSE) %>%
  filter(model == "rq1_primary", term == "paragraph_type_ai_vs_human")
rows[["s5_rq1"]] <- make_row("Study 5", "RQ1: AI vs human stance shift", "points (0-100)",
                             s5_rq1$ame, se_from_ci(s5_rq1$ame_low, s5_rq1$ame_high),
                             s5_assumed_main)

s5_rq2 <- read_csv(input_path("followup_persuasion", "rq2_ame.csv"), show_col_types = FALSE) %>%
  filter(model == "rq2_binned", term == "dose_response_ai_vs_human_high_minus_low")
rows[["s5_rq2"]] <- make_row("Study 5", "RQ2: dose-response (high minus low)", "points (0-100)",
                             s5_rq2$ame, se_from_ci(s5_rq2$ame_low, s5_rq2$ame_high),
                             s5_assumed_int)

mde_summary <- bind_rows(rows)
write_csv(mde_summary, file.path(RESULTS_DIR, "mde_summary.csv"))

# Study 1 attribute summary (median and range across 0-100 attributes).
s1_attr_summary <- mde_summary %>%
  filter(study == "Study 1", str_starts(contrast, "AI vs human")) %>%
  summarise(
    n_attributes = n(),
    mde80_median = median(mde80), mde80_min = min(mde80), mde80_max = max(mde80),
    abs_estimate_median = median(abs(estimate)),
    abs_estimate_min = min(abs(estimate)), abs_estimate_max = max(abs(estimate))
  )
write_csv(s1_attr_summary, file.path(RESULTS_DIR, "mde_study1_attribute_summary.csv"))

# =============================================================================
# PART B: REALISED VS ASSUMED VARIANCE COMPONENTS (STUDIES 4-5)
# =============================================================================

squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

re_sd <- function(model, group) {
  sqrt(as.numeric(VarCorr(model)$cond[[group]][1, 1]))
}

# --- Study 4: beta regression, reader random intercept (as in trust script) ---
trust <- read_csv("./data/followup_trust/annotations.csv", show_col_types = FALSE) %>%
  mutate(paragraph_type = relevel(factor(recode(source, human = "human", ai = "ai",
                                                .default = NA_character_)), ref = "human")) %>%
  filter(!is.na(trust_allocation_pence), !is.na(paragraph_type), !is.na(rater_id))

if (demo_mode) {
  set.seed(123)
  sampled <- sample(unique(trust$rater_id), min(500, n_distinct(trust$rater_id)))
  trust <- trust %>% filter(rater_id %in% sampled)
}

trust$y <- squeeze01(pmin(pmax(trust$trust_allocation_pence / 20, 0), 1))
m4 <- glmmTMB(y ~ paragraph_type + (1 | rater_id), data = trust,
              family = beta_family(link = "logit"))

# --- Study 5: linear mixed model, crossed reader + proposition intercepts ---
# Fit on treated cycles with the same stance_shift construction as the
# persuasion script, matching the treated-only data the power simulation used.
pers_raw <- read_csv("./data/followup_persuasion/annotations.csv", show_col_types = FALSE) %>%
  mutate(reader_id = as.character(rater_id),
         writer_id = as.character(writer_id),
         proposition_id = as.character(proposition_id))

if (demo_mode) {
  set.seed(123)
  sampled <- sample(unique(pers_raw$reader_id), min(1000, n_distinct(pers_raw$reader_id)))
  pers_raw <- pers_raw %>% filter(reader_id %in% sampled)
}

pers_pairs <- read_csv("./data/followup_persuasion/paragraph_pairs.csv", show_col_types = FALSE) %>%
  mutate(writer_id = as.character(writer_id), proposition_id = as.character(proposition_id)) %>%
  select(writer_id, proposition_id, writer_stance_human, writer_stance_ai) %>%
  pivot_longer(c(writer_stance_human, writer_stance_ai),
               names_to = "source", names_prefix = "writer_stance_", values_to = "writer_stance")

policy_attitude <- function(support, bad_idea, good_consequences, provided) {
  coalesce((support + (100 - bad_idea) + good_consequences) / 3, provided)
}

pers <- pers_raw %>%
  filter(condition_type != "static_control") %>%
  left_join(pers_pairs, by = c("writer_id", "proposition_id", "source")) %>%
  mutate(
    pre = policy_attitude(pre_support, pre_bad_idea, pre_good_consequences, policy_attitude_pre),
    post = policy_attitude(post_support, post_bad_idea, post_good_consequences, policy_attitude_post),
    direction = if_else(writer_stance < 50, -1, 1),
    stance_shift = direction * (post - pre),
    paragraph_type = factor(source, levels = c("human", "ai"))
  ) %>%
  filter(!is.na(stance_shift), !is.na(paragraph_type))

m5 <- glmmTMB(stance_shift ~ paragraph_type * distortion_bin + (1 | reader_id) + (1 | proposition_id),
              data = pers, family = gaussian())

variance_components <- tribble(
  ~study, ~component, ~scale, ~assumed, ~realised,
  "Study 4", "reader SD", "logit", s4_assumed_sd_reader, re_sd(m4, "rater_id"),
  "Study 4", "beta precision (phi)", "-", s4_assumed_phi, sigma(m4),
  "Study 5", "reader SD", "points (0-100)", s5_assumed_sd_reader, re_sd(m5, "reader_id"),
  "Study 5", "proposition SD", "points (0-100)", s5_assumed_sd_prop, re_sd(m5, "proposition_id"),
  "Study 5", "residual SD", "points (0-100)", s5_assumed_sd_resid, sigma(m5)
)
write_csv(variance_components, file.path(RESULTS_DIR, "variance_components.csv"))

# =============================================================================
# REPORT
# =============================================================================

options(width = 200)
cat("\n=== MDE80 (alpha = .05, two-sided) ===\n")
print(mde_summary %>% filter(!str_starts(contrast, "AI vs human:")), n = Inf)
cat("\n=== Study 1: 0-100 rating attributes ===\n")
print(s1_attr_summary)
cat("\n=== Variance components (Studies 4-5) ===\n")
print(variance_components)
cat("\nSaved to", RESULTS_DIR, "\n")
