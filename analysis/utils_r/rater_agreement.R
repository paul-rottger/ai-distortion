# Paragraph-level inter-rater agreement for Phase 2 annotations.
#
# Krippendorff's alpha (interval / ordinal / nominal metric by variable type)
# for all annotation attributes, plus the mean within-paragraph SD for scale
# attributes. Alpha is computed from unit-level summaries rather than a dense
# rater x unit matrix, which handles the sparse crossed rater design. Units with
# fewer than two ratings are not pairable and are dropped.
#
# Requires variable_definitions.R to be sourced first.

# Interval metric: per-unit count, sum, and sum of squares
interval_units <- function(unit_id, values) {
  tibble(unit_id = unit_id, v = values) %>%
    filter(!is.na(v)) %>%
    group_by(unit_id) %>%
    summarise(m = n(), s = sum(v), ss = sum(v^2), .groups = "drop") %>%
    filter(m >= 2) %>%
    mutate(ss_within = ss - s^2 / m)
}

interval_alpha <- function(units) {
  n <- sum(units$m)
  ss_total <- sum(units$ss) - sum(units$s)^2 / n
  observed <- sum(units$m / (units$m - 1) * units$ss_within)
  1 - (n - 1) * observed / (n * ss_total)
}

# Mean across paragraphs of the SD of ratings within each paragraph
mean_within_sd <- function(units) {
  mean(sqrt(units$ss_within / (units$m - 1)))
}

# Nominal / ordinal metric: per-unit category counts (units x categories)
categorical_units <- function(unit_id, values, levels) {
  counts <- tibble(unit_id = unit_id, v = factor(values, levels = levels)) %>%
    filter(!is.na(v)) %>%
    count(unit_id, v, .drop = FALSE) %>%
    pivot_wider(names_from = v, values_from = n, values_fill = 0)

  counts_matrix <- as.matrix(counts[, levels])
  m <- rowSums(counts_matrix)
  list(counts = counts_matrix[m >= 2, , drop = FALSE], m = m[m >= 2])
}

distance_matrix <- function(marginals, metric) {
  k <- length(marginals)
  if (metric == "nominal") {
    return(1 - diag(k))
  }
  # Krippendorff's ordinal distance, based on marginal category frequencies
  cum <- cumsum(marginals)
  outer(seq_len(k), seq_len(k), Vectorize(function(c, g) {
    lo <- min(c, g)
    hi <- max(c, g)
    (cum[hi] - (if (lo > 1) cum[lo - 1] else 0) - (marginals[c] + marginals[g]) / 2)^2
  }))
}

categorical_alpha <- function(units, metric) {
  marginals <- colSums(units$counts)
  n <- sum(marginals)
  delta <- distance_matrix(marginals, metric)
  observed <- sum(rowSums((units$counts %*% delta) * units$counts) / (units$m - 1))
  expected <- sum(outer(marginals, marginals) * delta)
  1 - (n - 1) * observed / expected
}

compute_agreement <- function(df, attribute, metric) {
  values <- df[[attribute]]

  if (metric == "interval") {
    units <- interval_units(df$unit_id, values)
    alpha <- interval_alpha(units)
    mean_sd <- mean_within_sd(units)
  } else {
    levels <- if (metric == "ordinal") ordinal_levels[[attribute]] else sort(unique(na.omit(values)))
    levels <- setdiff(levels, "Other")
    units <- categorical_units(df$unit_id, values, levels)
    alpha <- categorical_alpha(units, metric)
    mean_sd <- NA_real_
  }

  tibble(alpha = alpha, mean_sd = mean_sd)
}

# Computes agreement for all attributes, overall and by paragraph type, and
# writes alpha.csv (all attributes) and scale_mean_sd.csv (scale attributes)
# to results_dir (rows: attributes; columns: overall, writer, model, edited).
#
# Arguments:
#   data        - annotation rows with a unit_id column identifying paragraphs
#   results_dir - output directory
run_rater_agreement <- function(data, results_dir) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  # "Other" has no place on the education scale, so treat it as missing
  data <- data %>%
    mutate(writer_education = na_if(writer_education, "Other"))

  attribute_metrics <- bind_rows(
    tibble(attribute = rating_attributes, variable_type = "scale", metric = "interval"),
    tibble(attribute = ordinal_vars, variable_type = "ordinal", metric = "ordinal"),
    tibble(attribute = nominal_vars, variable_type = "nominal", metric = "nominal")
  )

  subsets <- c(
    list(overall = data),
    split(data, data$paragraph_type)[c("writer", "model", "edited")]
  )

  agreement <- pmap_dfr(attribute_metrics, function(attribute, variable_type, metric) {
    print(paste("computing agreement for:", attribute))
    imap_dfr(subsets, function(subset_df, subset_name) {
      tibble(
        attribute = attribute,
        variable_type = variable_type,
        subset = subset_name
      ) %>%
        bind_cols(compute_agreement(subset_df, attribute, metric))
    })
  })

  to_wide <- function(df, value_column) {
    df %>%
      select(attribute, subset, all_of(value_column)) %>%
      pivot_wider(names_from = subset, values_from = all_of(value_column)) %>%
      select(attribute, all_of(names(subsets)))
  }

  # Krippendorff's alpha for all 29 attributes
  alpha_table <- to_wide(agreement, "alpha")
  write_csv(alpha_table, file.path(results_dir, "alpha.csv"))
  print(alpha_table, n = Inf)

  # Mean within-paragraph SD for the 20 scale attributes
  sd_table <- to_wide(filter(agreement, variable_type == "scale"), "mean_sd")
  write_csv(sd_table, file.path(results_dir, "scale_mean_sd.csv"))
  print(sd_table, n = Inf)

  invisible(agreement)
}
