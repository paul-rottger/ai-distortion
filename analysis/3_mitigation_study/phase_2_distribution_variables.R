#!/usr/bin/env Rscript

# =============================================================================
# FOLLOWUP MITIGATION STUDY - PHASE 2 DISTRIBUTION ANALYSIS: ALL VARIABLES
#
# Compares writer and mitigation-conditioned model paragraph distributions
# across scale, ordinal, and nominal annotation variables.
#
# - Runs t-tests by paragraph type for scale outcomes.
# - Computes scale-attribute correlation matrices and heatmaps.
# - Runs Mann-Whitney U tests with Cliff's delta for ordinal outcomes.
# - Runs chi-squared tests with Cramer's V for nominal outcomes.
# - Runs analyses on unedited, edited, and preferred subsets.
# - Writes distribution result tables to results/followup_mitigation_phase_2_distribution/.
# - Saves scale correlation figures under figures/followup_mitigation_phase_2_distributions/.
#
# =============================================================================

# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== COMMAND-LINE FLAGS ----
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- get_results_dir(demo_mode, "followup_mitigation_phase_2_distribution")
FIGURES_DIR <- get_figures_dir(demo_mode, "followup_mitigation_phase_2_distributions")
DATA_SPLITS <- c("unedited", "edited", "preferred")
CORRELATION_ATTRIBUTES <- rating_attributes

# ===== DATA IMPORTS AND PROCESSING ----
scale_splits <- load_phase2_splits(
  "./data/followup_mitigation_phase_2/annotations.csv",
  "./data/followup_mitigation_phase_1/proposition_responses.csv"
)

prepare_categorical_split <- function(data) {
  data %>%
    filter(writer_education != "Other") %>%
    mutate(across(
      all_of(ordinal_vars),
      ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE)
    ))
}

categorical_splits <- list(
  unedited = prepare_categorical_split(scale_splits$data_unedited),
  edited = prepare_categorical_split(scale_splits$data_edited),
  preferred = prepare_categorical_split(scale_splits$data_preferred)
)

get_categorical_split_data <- function(data_split) {
  split_data <- categorical_splits[[data_split]]

  if (demo_mode) {
    split_data <- split_data %>%
      slice_sample(n = min(1000, nrow(split_data)))
  }

  split_data
}

get_scale_split_data <- function(data_split) {
  split_data <- scale_splits$get_split_data(data_split)

  if (demo_mode) {
    split_data <- split_data %>%
      slice_sample(n = min(1000, nrow(split_data)))
  }

  split_data
}

# ===== CORRELATION HELPERS ----
compute_correlation_results <- function(df, attributes) {
  pairs <- expand_grid(
    attribute_x = attributes,
    attribute_y = attributes
  )

  pair_results <- pmap_dfr(
    pairs,
    function(attribute_x, attribute_y) {
      pair_df <- df %>%
        select(all_of(attribute_x), all_of(attribute_y)) %>%
        drop_na()

      n_obs <- nrow(pair_df)

      if (n_obs < 3) {
        return(tibble(
          attribute_x = attribute_x,
          attribute_y = attribute_y,
          n = n_obs,
          correlation = NA_real_,
          p_value = NA_real_
        ))
      }

      x_vals <- pair_df[[attribute_x]]
      y_vals <- pair_df[[attribute_y]]

      if (sd(x_vals) == 0 || sd(y_vals) == 0) {
        return(tibble(
          attribute_x = attribute_x,
          attribute_y = attribute_y,
          n = n_obs,
          correlation = NA_real_,
          p_value = NA_real_
        ))
      }

      test_out <- cor.test(x_vals, y_vals, method = "pearson")

      tibble(
        attribute_x = attribute_x,
        attribute_y = attribute_y,
        n = n_obs,
        correlation = unname(test_out$estimate),
        p_value = test_out$p.value
      )
    }
  )

  pair_results %>%
    mutate(
      label_x = attribute_x,
      label_y = attribute_y,
      label_text = case_when(
        !is.na(p_value) & p_value < 0.001 ~ sprintf("%.2f", correlation),
        TRUE ~ ""
      )
    )
}

create_correlation_heatmap <- function(correlation_results, data_split) {
  figure_dir <- file.path(FIGURES_DIR, data_split)
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

  plot_data <- correlation_results %>%
    mutate(
      correlation = if_else(!is.na(p_value) & p_value < 0.001, correlation, NA_real_),
      label_x = factor(label_x, levels = CORRELATION_ATTRIBUTES),
      label_y = factor(label_y, levels = rev(CORRELATION_ATTRIBUTES))
    )

  heatmap_plot <- ggplot(plot_data, aes(x = label_x, y = label_y, fill = correlation)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = label_text), size = 2.7, na.rm = FALSE) +
    scale_fill_gradient2(
      low = "#2166ac",
      mid = "#f7f7f7",
      high = "#b2182b",
      midpoint = 0,
      limits = c(-1, 1),
      na.value = "#d9d9d9",
      name = "Pearson r"
    ) +
    coord_fixed() +
    labs(
      title = paste("Scale Attribute Correlations:", str_to_title(data_split)),
      x = NULL,
      y = NULL,
      caption = "Cells show Pearson correlation coefficients rounded to two decimals. Cells are masked unless p < .001."
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      axis.text.y = element_text(hjust = 1),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold"),
      plot.caption = element_text(hjust = 0),
      legend.position = "right"
    )

  ggsave(
    filename = file.path(figure_dir, "scale_attribute_correlation_matrix.pdf"),
    plot = heatmap_plot,
    width = 14,
    height = 12,
    dpi = 300
  )
}

# ===== SCALE TESTS ----
run_scale_test_by_type <- function(df, attribute) {
  df_sub <- df %>%
    select(paragraph_type_, all_of(attribute)) %>%
    drop_na()

  writer_vals <- df_sub[df_sub$paragraph_type_ == "writer", attribute, drop = TRUE]
  model_vals <- df_sub[df_sub$paragraph_type_ == "model", attribute, drop = TRUE]

  t_out <- t.test(model_vals, writer_vals)

  pooled_sd <- sqrt(
    ((length(model_vals) - 1) * var(model_vals) +
       (length(writer_vals) - 1) * var(writer_vals)) /
      (length(model_vals) + length(writer_vals) - 2)
  )

  cohens_d <- (mean(model_vals) - mean(writer_vals)) / pooled_sd

  tibble(
    term = "paragraph_type_model",
    mean_writer = mean(writer_vals),
    mean_model = mean(model_vals),
    mean_difference = mean(model_vals) - mean(writer_vals),
    ci_low = t_out$conf.int[1],
    ci_high = t_out$conf.int[2],
    t_statistic = unname(t_out$statistic),
    p_value = t_out$p.value,
    cohens_d = cohens_d
  )
}

run_all_scale_tests <- function() {
  for (attribute in rating_attributes) {
    message("Running scale tests for: ", attribute)

    for (data_split in DATA_SPLITS) {
      dir.create(
        file.path(RESULTS_DIR, data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- run_scale_test_by_type(
        df = get_scale_split_data(data_split),
        attribute = attribute
      )

      write_csv(
        results,
        file.path(RESULTS_DIR, data_split, paste0(attribute, "_by_type.csv"))
      )
    }
  }
}

run_scale_correlations <- function() {
  for (data_split in DATA_SPLITS) {
    message("Running scale correlations for: ", data_split)

    split_data <- get_scale_split_data(data_split)

    dir.create(
      file.path(RESULTS_DIR, data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    correlation_results <- compute_correlation_results(
      df = split_data,
      attributes = CORRELATION_ATTRIBUTES
    )

    write_csv(
      correlation_results,
      file.path(RESULTS_DIR, data_split, "scale_attribute_correlations.csv")
    )

    create_correlation_heatmap(
      correlation_results = correlation_results,
      data_split = data_split
    )
  }
}

# ===== ORDINAL TESTS ----
run_ordinal_test <- function(data, ordinal_var) {
  x <- as.numeric(data[[ordinal_var]])
  g <- data$paragraph_type_

  mw <- wilcox.test(x ~ g, exact = FALSE)
  cd <- cliff.delta(x ~ g)

  list(
    test = mw,
    cliffs_delta = cd$estimate,
    cliffs_delta_ci_low = cd$conf.int[1],
    cliffs_delta_ci_high = cd$conf.int[2]
  )
}

run_all_ordinal_tests <- function() {
  for (attribute in ordinal_vars) {
    message("Running ordinal tests for: ", attribute)

    for (data_split in DATA_SPLITS) {
      dir.create(
        file.path(RESULTS_DIR, data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- run_ordinal_test(get_categorical_split_data(data_split), attribute)

      write_csv(
        tibble(
          term = "paragraph_type_model",
          statistic = unname(results$test$statistic),
          p_value = results$test$p.value,
          cliffs_delta = -results$cliffs_delta,
          cliffs_delta_ci_high = -results$cliffs_delta_ci_low,
          cliffs_delta_ci_low = -results$cliffs_delta_ci_high
        ),
        file.path(RESULTS_DIR, data_split, paste0(attribute, "_by_type.csv"))
      )
    }
  }
}

# ===== NOMINAL TESTS ----
cramers_v <- function(data, rating_attribute) {
  tab <- table(data$paragraph_type_, data[[rating_attribute]])

  if (nrow(tab) < 2 || ncol(tab) < 2) {
    return(NA_real_)
  }

  chisq <- suppressWarnings(chisq.test(tab, correct = FALSE))
  n <- sum(tab)
  dims <- dim(tab)

  sqrt(as.numeric(chisq$statistic) / (n * (min(dims) - 1)))
}

bootstrap_cramers_v_ci <- function(data, rating_attribute, R = 1000, conf = 0.95) {
  v_boot <- replicate(R, {
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    cramers_v(data[idx, ], rating_attribute)
  })

  alpha <- (1 - conf) / 2
  quantile(v_boot, c(alpha, 1 - alpha), na.rm = TRUE)
}

run_nominal_test <- function(data, rating_attribute) {
  tab <- table(data$paragraph_type_, data[[rating_attribute]])
  chisq <- suppressWarnings(chisq.test(tab))

  n <- sum(tab)
  dims <- dim(tab)
  cramers_v_value <- sqrt(as.numeric(chisq$statistic) / (n * (min(dims) - 1)))

  cramers_v_ci <- bootstrap_cramers_v_ci(data, rating_attribute)

  list(
    test = chisq,
    cramers_v = cramers_v_value,
    cramers_v_ci_low = cramers_v_ci[1],
    cramers_v_ci_high = cramers_v_ci[2]
  )
}

run_all_nominal_tests <- function() {
  for (attribute in nominal_vars) {
    message("Running nominal tests for: ", attribute)

    for (data_split in DATA_SPLITS) {
      dir.create(
        file.path(RESULTS_DIR, data_split),
        recursive = TRUE,
        showWarnings = FALSE
      )

      results <- run_nominal_test(get_categorical_split_data(data_split), attribute)

      write_csv(
        tibble(
          term = "paragraph_type_model",
          statistic = unname(results$test$statistic),
          p_value = results$test$p.value,
          cramers_v = results$cramers_v,
          cramers_v_ci_low = results$cramers_v_ci_low,
          cramers_v_ci_high = results$cramers_v_ci_high
        ),
        file.path(RESULTS_DIR, data_split, paste0(attribute, "_by_type.csv"))
      )
    }
  }
}

if (demo_mode) {
  message("Running in demo mode on n=1000 samples from each data split.")
}

run_all_scale_tests()
run_scale_correlations()
run_all_ordinal_tests()
run_all_nominal_tests()