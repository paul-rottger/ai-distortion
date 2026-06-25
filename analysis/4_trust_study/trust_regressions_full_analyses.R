#!/usr/bin/env Rscript

# =============================================================================
# FOLLOWUP TRUST STUDY - FULL BETA REGRESSION ANALYSES
#
# Implements:
# - RQ1 primary: trust_allocation ~ paragraph_type + (1 | reader_id)
# - RQ1 + stance distance control
# - RQ1 + presentation order control
# - RQ2 interaction: paragraph_type * distortion_bin
# - RQ2 continuous moderation: paragraph_type * distortion_magnitude
# - Exploratory: separate models adding each of the 20 paragraph attributes
#
# Outcome is trust_allocation_pence (0-20), modeled via beta regression with
# logit link after rescaling to [0,1] and squeezing to (0,1).
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(glmmTMB)
  library(broom.mixed)
  library(marginaleffects)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/demo_paths.R")

set.seed(123)

demo_mode <- parse_demo_mode()

INPUT_TRUST_PATH <- "./data/followup_trust/annotations.csv"
INPUT_PAIRS_PATH <- "./data/followup_trust/paragraph_pairs.csv"
RESULTS_DIR <- get_results_dir(demo_mode, "followup_trust")
FIGURES_DIR <- get_figures_dir(demo_mode, "followup_trust")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# HELPERS
# =============================================================================

require_columns <- function(df, cols, label) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(
      paste0(
        "Missing required columns in ",
        label,
        ": ",
        paste(missing, collapse = ", ")
      )
    )
  }
}

# Helper function: squeeze 0..1 for beta regression
# Source: https://pubmed.ncbi.nlm.nih.gov/16594767/
squeeze01 <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

fit_beta_model <- function(df, formula_rhs, model_name) {
  y <- df$trust_allocation_pence / 20
  y <- pmin(pmax(y, 0), 1)
  y <- squeeze01(y)

  form <- as.formula(paste0("y ~ ", formula_rhs))
  model <- glmmTMB(
    form,
    data = df,
    family = beta_family(link = "logit")
  )

  fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
    mutate(
      model = model_name,
      odds_ratio = exp(estimate),
      or_low = exp(conf.low),
      or_high = exp(conf.high),
      p = p.value
    ) %>%
    select(model, term, estimate, conf.low, conf.high, statistic, p, odds_ratio, or_low, or_high)

  list(model = model, fixed = fixed)
}

compute_ame_contrast <- function(model, variable, model_name, by_var = NULL) {
  args <- list(
    model = model,
    variables = setNames(list("reference"), variable),
    type = "response",
    re.form = NA
  )

  if (!is.null(by_var)) {
    args$by <- by_var
  }

  out <- do.call(avg_comparisons, args) %>%
    as_tibble() %>%
    mutate(
      model = model_name,
      term = paste0(variable, "_", gsub(" - ", "_vs_", contrast)),
      ame_pence = estimate * 20,
      ame_low_pence = conf.low * 20,
      ame_high_pence = conf.high * 20,
      p = p.value
    )

  keep <- c("model", "term", "ame_pence", "ame_low_pence", "ame_high_pence", "statistic", "p")
  if (!is.null(by_var) && by_var %in% names(out)) {
    keep <- c("model", by_var, "term", "ame_pence", "ame_low_pence", "ame_high_pence", "statistic", "p")
  }

  out %>% select(all_of(keep))
}

compute_ame_slope <- function(model, variable, model_name) {
  avg_slopes(
    model,
    variables = variable,
    type = "response",
    re.form = NA
  ) %>%
    as_tibble() %>%
    mutate(
      model = model_name,
      term = variable,
      ame_pence = estimate * 20,
      ame_low_pence = conf.low * 20,
      ame_high_pence = conf.high * 20,
      p = p.value
    ) %>%
    select(model, term, ame_pence, ame_low_pence, ame_high_pence, statistic, p)
}

# =============================================================================
# DATA PREP
# =============================================================================

trust <- read_csv(INPUT_TRUST_PATH, show_col_types = FALSE) %>%
  mutate(
    row_in_file = row_number(),
    writer_id = as.character(writer_id),
    proposition_id = as.character(proposition_id),
    paragraph_type = recode(source, human = "human", ai = "ai", .default = NA_character_),
    paragraph_type = relevel(factor(paragraph_type), ref = "human"),
    distortion_bin = relevel(factor(distortion_bin), ref = "low_distortion")
  ) %>%
  group_by(rater_id) %>%
  arrange(row_in_file, .by_group = TRUE) %>%
  mutate(presentation_order = row_number()) %>%
  ungroup()

# Demo mode: sub-sample raters (keep all rows per sampled rater).
if (demo_mode) {
  sampled_raters <- trust %>%
    distinct(rater_id) %>%
    slice_sample(n = min(500, n_distinct(trust$rater_id))) %>%
    pull(rater_id)
  trust <- trust %>% filter(rater_id %in% sampled_raters)
  message("Running in demo mode on ", length(sampled_raters), " raters.")
}

if (!("human" %in% levels(trust$paragraph_type))) {
  stop("Expected paragraph_type level 'human' was not found.")
}
if (!("low_distortion" %in% levels(trust$distortion_bin))) {
  stop("Expected distortion_bin level 'low_distortion' was not found.")
}

pairs <- read_csv(INPUT_PAIRS_PATH, show_col_types = FALSE) %>%
  mutate(
    writer_id = as.character(writer_id),
    proposition_id = as.character(proposition_id)
  )

delta_cols <- paste0("delta_", rating_attributes)
require_columns(
  pairs,
  c("writer_id", "proposition_id", "writer_stance_human", "writer_stance_ai", delta_cols),
  "pairs_with_deltas.csv"
)

human_cols <- paste0(rating_attributes, "_human")
ai_cols <- paste0(rating_attributes, "_ai")
require_columns(
  pairs,
  c(human_cols, ai_cols),
  "pairs_with_deltas.csv"
)

pairs <- pairs %>%
  mutate(
    distortion_magnitude = rowMeans(across(all_of(delta_cols), ~ abs(.x)), na.rm = TRUE)
  )

pairs_long <- pairs %>%
  select(
    writer_id,
    proposition_id,
    distortion_magnitude,
    writer_stance_human,
    writer_stance_ai,
    all_of(human_cols),
    all_of(ai_cols)
  ) %>%
  pivot_longer(
    cols = -c(writer_id, proposition_id, distortion_magnitude),
    names_to = c(".value", "paragraph_type"),
    names_pattern = "(.*)_(human|ai)"
  )

analysis_data <- trust %>%
  left_join(pairs_long, by = c("writer_id", "proposition_id", "paragraph_type")) %>%
  mutate(
    paragraph_type = relevel(factor(paragraph_type, levels = c("human", "ai")), ref = "human"),
    distortion_bin = relevel(factor(distortion_bin, levels = c("low_distortion", "high_distortion")), ref = "low_distortion"),
    stance_distance = abs(rater_stance_post - writer_stance)
  )

required_core <- c(
  "trust_allocation_pence",
  "paragraph_type",
  "distortion_bin",
  "distortion_magnitude",
  "presentation_order",
  "stance_distance",
  "rater_id"
)
require_columns(analysis_data, required_core, "analysis_data")

# =============================================================================
# RQ1 MODELS
# =============================================================================

rq1_primary_df <- analysis_data %>%
  filter(!is.na(trust_allocation_pence), !is.na(paragraph_type), !is.na(rater_id))

rq1_primary <- fit_beta_model(
  rq1_primary_df,
  "paragraph_type + (1 | rater_id)",
  "rq1_primary"
)

rq1_primary_ame <- compute_ame_contrast(
  rq1_primary$model,
  variable = "paragraph_type",
  model_name = "rq1_primary"
)

rq1_stance_df <- analysis_data %>%
  filter(
    !is.na(trust_allocation_pence),
    !is.na(paragraph_type),
    !is.na(stance_distance),
    !is.na(rater_id)
  )

rq1_with_stance <- fit_beta_model(
  rq1_stance_df,
  "paragraph_type + stance_distance + (1 | rater_id)",
  "rq1_with_stance_distance"
)

rq1_with_stance_ame <- bind_rows(
  compute_ame_contrast(
    rq1_with_stance$model,
    variable = "paragraph_type",
    model_name = "rq1_with_stance_distance"
  ),
  compute_ame_slope(
    rq1_with_stance$model,
    variable = "stance_distance",
    model_name = "rq1_with_stance_distance"
  )
)

rq1_order_df <- analysis_data %>%
  filter(
    !is.na(trust_allocation_pence),
    !is.na(paragraph_type),
    !is.na(presentation_order),
    !is.na(rater_id)
  )

rq1_with_order <- fit_beta_model(
  rq1_order_df,
  "paragraph_type + presentation_order + (1 | rater_id)",
  "rq1_with_presentation_order"
)

rq1_with_order_ame <- bind_rows(
  compute_ame_contrast(
    rq1_with_order$model,
    variable = "paragraph_type",
    model_name = "rq1_with_presentation_order"
  ),
  compute_ame_slope(
    rq1_with_order$model,
    variable = "presentation_order",
    model_name = "rq1_with_presentation_order"
  )
)

# =============================================================================
# RQ2 MODELS
# =============================================================================

rq2_interaction_df <- analysis_data %>%
  filter(
    !is.na(trust_allocation_pence),
    !is.na(paragraph_type),
    !is.na(distortion_bin),
    !is.na(rater_id)
  )

rq2_interaction <- fit_beta_model(
  rq2_interaction_df,
  "paragraph_type * distortion_bin + (1 | rater_id)",
  "rq2_interaction_bin"
)

rq2_interaction_ame <- compute_ame_contrast(
  rq2_interaction$model,
  variable = "paragraph_type",
  model_name = "rq2_interaction_bin",
  by_var = "distortion_bin"
)

rq2_continuous_df <- analysis_data %>%
  filter(
    !is.na(trust_allocation_pence),
    !is.na(paragraph_type),
    !is.na(distortion_magnitude),
    !is.na(rater_id)
  )

rq2_continuous <- fit_beta_model(
  rq2_continuous_df,
  "paragraph_type * distortion_magnitude + (1 | rater_id)",
  "rq2_interaction_continuous"
)

rq2_distortion_mean <- mean(rq2_continuous_df$distortion_magnitude, na.rm = TRUE)
rq2_distortion_sd <- sd(rq2_continuous_df$distortion_magnitude, na.rm = TRUE)

rq2_continuous_growth_grid_1 <- datagrid(
  model = rq2_continuous$model,
  distortion_magnitude = c(rq2_distortion_mean, rq2_distortion_mean + 1)
)
rq2_continuous_growth_contrast_1 <- comparisons(
  rq2_continuous$model,
  variables = list(paragraph_type = c("human", "ai")),
  newdata = rq2_continuous_growth_grid_1,
  type = "response",
  re.form = NA
)
rq2_continuous_growth_per_1 <- hypotheses(
  rq2_continuous_growth_contrast_1,
  hypothesis = "b2 - b1 = 0"
) %>%
  as_tibble() %>%
  transmute(
    model = "rq2_interaction_continuous",
    term = "ai_advantage_growth_per_1_distortion",
    ame_pence = estimate * 20,
    ame_low_pence = conf.low * 20,
    ame_high_pence = conf.high * 20,
    statistic,
    p = p.value
  )

rq2_continuous_growth_grid_1sd <- datagrid(
  model = rq2_continuous$model,
  distortion_magnitude = c(rq2_distortion_mean, rq2_distortion_mean + rq2_distortion_sd)
)
rq2_continuous_growth_contrast_1sd <- comparisons(
  rq2_continuous$model,
  variables = list(paragraph_type = c("human", "ai")),
  newdata = rq2_continuous_growth_grid_1sd,
  type = "response",
  re.form = NA
)
rq2_continuous_growth_per_1sd <- hypotheses(
  rq2_continuous_growth_contrast_1sd,
  hypothesis = "b2 - b1 = 0"
) %>%
  as_tibble() %>%
  transmute(
    model = "rq2_interaction_continuous",
    term = "ai_advantage_growth_per_1sd_distortion",
    ame_pence = estimate * 20,
    ame_low_pence = conf.low * 20,
    ame_high_pence = conf.high * 20,
    statistic,
    p = p.value
  )

rq2_continuous_ai_advantage_growth <- bind_rows(
  rq2_continuous_growth_per_1,
  rq2_continuous_growth_per_1sd
)

rq2_continuous_ame <- bind_rows(
  compute_ame_contrast(
    rq2_continuous$model,
    variable = "paragraph_type",
    model_name = "rq2_interaction_continuous"
  ),
  compute_ame_slope(
    rq2_continuous$model,
    variable = "distortion_magnitude",
    model_name = "rq2_interaction_continuous"
  ),
  rq2_continuous_ai_advantage_growth
)

# =============================================================================
# EXPLORATORY: ATTRIBUTE-BY-ATTRIBUTE MODELS
# =============================================================================

exploratory_fixed <- list()
exploratory_ame <- list()

for (attr in rating_attributes) {
  model_name <- paste0("exploratory_", attr)
  
  print(paste0("Fitting model for attribute: ", attr))

  df_attr <- analysis_data %>%
    filter(
      !is.na(trust_allocation_pence),
      !is.na(paragraph_type),
      !is.na(.data[[attr]]),
      !is.na(rater_id)
    )

  fitted <- fit_beta_model(
    df_attr,
    paste0("paragraph_type + ", attr, " + (1 | rater_id)"),
    model_name
  )

  exploratory_fixed[[attr]] <- fitted$fixed

  attr_ame <- compute_ame_slope(
    fitted$model,
    variable = attr,
    model_name = model_name
  ) %>%
    mutate(attribute = attr) %>%
    select(attribute, everything())

  exploratory_ame[[attr]] <- attr_ame
}

exploratory_fixed_tbl <- bind_rows(exploratory_fixed)
exploratory_ame_tbl <- bind_rows(exploratory_ame)

# Bonferroni correction across the 20 exploratory attribute tests.
attr_term_mask <- exploratory_fixed_tbl$term %in% rating_attributes
exploratory_fixed_tbl$p_bonferroni <- NA_real_
exploratory_fixed_tbl$p_bonferroni[attr_term_mask] <- p.adjust(
  exploratory_fixed_tbl$p[attr_term_mask],
  method = "bonferroni"
)
exploratory_fixed_tbl$significant_bonferroni <- !is.na(exploratory_fixed_tbl$p_bonferroni) &
  exploratory_fixed_tbl$p_bonferroni < 0.05

exploratory_ame_tbl <- exploratory_ame_tbl %>%
  mutate(
    p_bonferroni = p.adjust(p, method = "bonferroni"),
    significant_bonferroni = p_bonferroni < 0.05
  )

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

write_csv(bind_rows(
  rq1_primary$fixed,
  rq1_with_stance$fixed,
  rq1_with_order$fixed
), file.path(RESULTS_DIR, "rq1_fixed_effects.csv"))

write_csv(bind_rows(
  rq1_primary_ame,
  rq1_with_stance_ame,
  rq1_with_order_ame
), file.path(RESULTS_DIR, "rq1_ame.csv"))

write_csv(bind_rows(
  rq2_interaction$fixed,
  rq2_continuous$fixed
), file.path(RESULTS_DIR, "rq2_fixed_effects.csv"))

write_csv(bind_rows(
  rq2_interaction_ame,
  rq2_continuous_ame
), file.path(RESULTS_DIR, "rq2_ame.csv"))

write_csv(
  rq2_continuous_ai_advantage_growth,
  file.path(RESULTS_DIR, "rq2_continuous_ai_advantage_growth.csv")
)

write_csv(
  exploratory_fixed_tbl,
  file.path(RESULTS_DIR, "exploratory_attributes_fixed_effects.csv")
)

write_csv(
  exploratory_ame_tbl,
  file.path(RESULTS_DIR, "exploratory_attributes_ame.csv")
)

coverage <- analysis_data %>%
  summarise(
    n_rows = n(),
    n_missing_stance_distance = sum(is.na(stance_distance)),
    n_missing_distortion_magnitude = sum(is.na(distortion_magnitude)),
    n_missing_writer_stance = sum(is.na(writer_stance)),
    n_missing_any_attribute = sum(if_any(all_of(rating_attributes), is.na))
  )

write_csv(coverage, file.path(RESULTS_DIR, "data_join_coverage.csv"))

message("Saved outputs to: ", RESULTS_DIR)
