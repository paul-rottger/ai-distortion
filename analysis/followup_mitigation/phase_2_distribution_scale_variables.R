# ===== PACKAGES ----
suppressPackageStartupMessages({
  library(tidyverse)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- "./results/followup_mitigation_phase_2_distribution"
FIGURES_DIR <- "./figures/followup_mitigation_phase_2_distributions"
DATA_SPLITS <- c("unedited", "edited", "preferred")

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
  "./data/followup_mitigation_phase_2/annotations.csv",
  "./data/followup_mitigation_phase_1/proposition_responses.csv"
), envir = environment())

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
      label_x = factor(label_x, levels = correlation_attributes),
      label_y = factor(label_y, levels = rev(correlation_attributes))
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

# ===== SIMPLE T-TEST FUNCTION ----

run_test_by_type <- function(df, attribute) {
  
  message("Running test for: ", attribute)
  
  df_sub <- df %>%
    select(paragraph_type_, all_of(attribute)) %>%
    drop_na()
  
  writer_vals <- df_sub %>%
    filter(paragraph_type_ == "writer") %>%
    pull(attribute)
  
  model_vals <- df_sub %>%
    filter(paragraph_type_ == "model") %>%
    pull(attribute)
  
  # Independent t-test
  t_out <- t.test(model_vals, writer_vals)
  
  # Cohen's d
  pooled_sd <- sqrt(
    ((length(model_vals) - 1) * var(model_vals) +
       (length(writer_vals) - 1) * var(writer_vals)) /
      (length(model_vals) + length(writer_vals) - 2)
  )
  
  d <- (mean(model_vals) - mean(writer_vals)) / pooled_sd
  
  tibble(
    term = "paragraph_type_model",
    mean_writer = mean(writer_vals),
    mean_model = mean(model_vals),
    mean_difference = mean(model_vals) - mean(writer_vals),
    ci_low = t_out$conf.int[1],
    ci_high = t_out$conf.int[2],
    t_statistic = t_out$statistic,
    p_value = t_out$p.value,
    cohens_d = d
  )
}

# ===== ATTRIBUTES ----

correlation_attributes <- rating_attributes

# ===== RUN ALL TESTS ----

for (attr in rating_attributes) {
  message("Running tests for: ", attr)

  for (data_split in DATA_SPLITS) {
    dir.create(
      file.path(RESULTS_DIR, data_split),
      recursive = TRUE,
      showWarnings = FALSE
    )

    results <- run_test_by_type(
      df = get_split_data(data_split),
      attribute = attr
    )

    write_csv(
      results,
      file.path(RESULTS_DIR, data_split, paste0(attr, "_by_type.csv"))
    )
  }
}

# ===== RUN CORRELATION ANALYSIS ----

for (data_split in DATA_SPLITS) {
  message("Running correlation analysis for: ", data_split)

  split_data <- get_split_data(data_split)

  dir.create(
    file.path(RESULTS_DIR, data_split),
    recursive = TRUE,
    showWarnings = FALSE
  )

  correlation_results <- compute_correlation_results(
    df = split_data,
    attributes = correlation_attributes
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