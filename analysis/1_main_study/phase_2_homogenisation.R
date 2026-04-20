#!/usr/bin/env Rscript

# =============================================================================
# MAIN STUDY - PHASE 2 HOMOGENISATION ANALYSIS
# 
# Quantifies homogenisation by comparing response spread across writer and model text.
#
# - Computes spread metrics for scale and non-scale outcomes.
# - Estimates bootstrap uncertainty and Bonferroni-adjusted significance.
# - Produces distributional comparison plots for configured data subsets.
# - Writes homogenisation outputs to main_phase_2_distribution results and figures directories.
# 
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# Set random seed for reproducibility
# ===== RANDOM SEED ----
set.seed(123)

# ===== ANALYSIS CONFIG ----
RESULTS_DIR <- "./results/main_phase_2_distribution"
FIGURES_DIR <- "./figures/main_phase_2_distributions"
DATA_SPLITS <- c(
	# "unedited",
	# "edited",
	"preferred"
)

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
	"./data/main_phase_2/annotations.csv",
	"./data/main_phase_1/proposition_responses.csv",
	extra_mutate = function(data) {
		data %>%
			mutate(across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE)))
	}
), envir = environment())

# ===== ATTRIBUTE LISTS ----
scale_attributes <- c(
	"paragraph_formality",
	"paragraph_clarity",
	"paragraph_informativeness",
	"paragraph_originality",
	"paragraph_relevance",
	"writer_knowledge",
	"writer_importance",
	"writer_confidence",
	"writer_stance_polarity",
	"paragraph_hope",
	"paragraph_excitement",
	"paragraph_fear",
	"paragraph_disgust",
	"paragraph_anger",
	"writer_affect_x",
	"writer_affect_y",
	"writer_optimism",
	"writer_community",
	"writer_friendliness",
	"writer_openness"
)

ordinal_attributes <- c(
	"writer_income",
	"writer_age_binned",
	"writer_english_first",
	"writer_english_skills",
	"writer_education"
)

nominal_attributes <- c(
	"writer_gender",
	"writer_race",
	"writer_politicalParty",
	"writer_politicalIdeology"
)

all_attributes <- c(scale_attributes, ordinal_attributes, nominal_attributes)
n_tests <- length(all_attributes)

# =============================================================================
# ANALYSIS SETUP
# =============================================================================

# ===== BOOTSTRAP SETTINGS ----
n_boot <- 1000

# ===== SPREAD METRICS + BOOTSTRAP TESTS ----
entropy_base2 <- function(x) {
	x <- x[!is.na(x)]

	if (length(x) == 0) {
		return(NA_real_)
	}

	probs <- prop.table(table(x))
	-sum(probs * log2(probs))
}

bootstrap_significant_bonf <- function(boot_stats, alpha, n_tests) {
	boot_stats <- boot_stats[is.finite(boot_stats)]

	if (length(boot_stats) == 0) {
		return(NA)
	}

	alpha_adj <- alpha / n_tests
	ci_low <- quantile(boot_stats, probs = alpha_adj / 2, na.rm = TRUE)
	ci_high <- quantile(boot_stats, probs = 1 - alpha_adj / 2, na.rm = TRUE)
	(ci_low > 0) || (ci_high < 0)
}

bootstrap_stat_diff <- function(writer_vals, model_vals, stat_fun, n_boot = 1000, n_tests = 1) {
	obs_writer <- stat_fun(writer_vals)
	obs_model <- stat_fun(model_vals)
	obs_diff <- obs_model - obs_writer

	boot_diffs <- replicate(n_boot, {
		writer_boot <- sample(writer_vals, size = length(writer_vals), replace = TRUE)
		model_boot <- sample(model_vals, size = length(model_vals), replace = TRUE)
		stat_fun(model_boot) - stat_fun(writer_boot)
	})

	tibble(
		group1_stat = obs_writer,
		group2_stat = obs_model,
		stat_difference = obs_diff,
		ci_low = quantile(boot_diffs, 0.025, na.rm = TRUE),
		ci_high = quantile(boot_diffs, 0.975, na.rm = TRUE),
		significant_0_05_bonferroni = bootstrap_significant_bonf(boot_diffs, 0.05, n_tests),
		significant_0_01_bonferroni = bootstrap_significant_bonf(boot_diffs, 0.01, n_tests),
		significant_0_001_bonferroni = bootstrap_significant_bonf(boot_diffs, 0.001, n_tests)
	)
}

bootstrap_test_attribute <- function(df, attribute, attribute_type, n_boot = 1000) {
	df_sub <- df %>%
		select(paragraph_type_, all_of(attribute)) %>%
		filter(!is.na(paragraph_type_), !is.na(.data[[attribute]])) %>%
		filter(paragraph_type_ %in% c("writer", "model"))

	if (nrow(df_sub) == 0) {
		return(tibble(
			attribute = attribute,
			attribute_type = attribute_type,
			spread_metric = NA_character_,
			spread_writer = NA_real_,
			spread_model = NA_real_,
			spread_difference = NA_real_,
			relative_dispersion_ratio = NA_real_,
			normalised_entropy_difference = NA_real_,
			ci_low = NA_real_,
			ci_high = NA_real_,
			significant_0_05_bonferroni = NA,
			significant_0_01_bonferroni = NA,
			significant_0_001_bonferroni = NA
		))
	}

	writer_vals <- df_sub %>%
		filter(paragraph_type_ == "writer") %>%
		pull(attribute)

	model_vals <- df_sub %>%
		filter(paragraph_type_ == "model") %>%
		pull(attribute)

	if (length(writer_vals) == 0 || length(model_vals) == 0) {
		return(tibble(
			attribute = attribute,
			attribute_type = attribute_type,
			spread_metric = NA_character_,
			spread_writer = NA_real_,
			spread_model = NA_real_,
			spread_difference = NA_real_,
			relative_dispersion_ratio = NA_real_,
			normalised_entropy_difference = NA_real_,
			ci_low = NA_real_,
			ci_high = NA_real_,
			significant_0_05_bonferroni = NA,
			significant_0_01_bonferroni = NA,
			significant_0_001_bonferroni = NA
		))
	}

	if (attribute_type == "scale") {
		writer_vals <- suppressWarnings(as.numeric(writer_vals))
		model_vals <- suppressWarnings(as.numeric(model_vals))
		writer_vals <- writer_vals[is.finite(writer_vals)]
		model_vals <- model_vals[is.finite(model_vals)]

		if (length(writer_vals) < 2 || length(model_vals) < 2) {
			return(tibble(
				attribute = attribute,
				attribute_type = attribute_type,
				spread_metric = "sd",
				spread_writer = NA_real_,
				spread_model = NA_real_,
				spread_difference = NA_real_,
				relative_dispersion_ratio = NA_real_,
				normalised_entropy_difference = NA_real_,
				ci_low = NA_real_,
				ci_high = NA_real_,
				significant_0_05_bonferroni = NA,
				significant_0_01_bonferroni = NA,
				significant_0_001_bonferroni = NA
			))
		}

		res <- bootstrap_stat_diff(writer_vals, model_vals, sd, n_boot = n_boot, n_tests = n_tests)
		return(res %>% transmute(
			attribute = attribute,
			attribute_type = attribute_type,
			spread_metric = "sd",
			spread_writer = group1_stat,
			spread_model = group2_stat,
			spread_difference = stat_difference,
			relative_dispersion_ratio = if_else(group1_stat > 0, group2_stat / group1_stat, NA_real_),
			normalised_entropy_difference = NA_real_,
			ci_low = ci_low,
			ci_high = ci_high,
			significant_0_05_bonferroni = significant_0_05_bonferroni,
			significant_0_01_bonferroni = significant_0_01_bonferroni,
			significant_0_001_bonferroni = significant_0_001_bonferroni
		))
	}

	res <- bootstrap_stat_diff(writer_vals, model_vals, entropy_base2, n_boot = n_boot, n_tests = n_tests)
	max_entropy <- {
		k <- n_distinct(c(writer_vals, model_vals))
		if (k > 1) {
			log2(k)
		} else {
			NA_real_
		}
	}

	res %>% transmute(
		attribute = attribute,
		attribute_type = attribute_type,
		spread_metric = "entropy_base2",
		spread_writer = group1_stat,
		spread_model = group2_stat,
		spread_difference = stat_difference,
		relative_dispersion_ratio = NA_real_,
		normalised_entropy_difference = if_else(is.finite(max_entropy) & max_entropy > 0, stat_difference / max_entropy, NA_real_),
		ci_low = ci_low,
		ci_high = ci_high,
		significant_0_05_bonferroni = significant_0_05_bonferroni,
		significant_0_01_bonferroni = significant_0_01_bonferroni,
		significant_0_001_bonferroni = significant_0_001_bonferroni
	)
}

create_scale_spread_boxplot <- function(df, spread_results, data_split) {
	figure_dir <- file.path(FIGURES_DIR, data_split)
	dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

	scale_spread_results <- spread_results %>%
		filter(attribute_type == "scale", spread_metric == "sd") %>%
		arrange(desc(relative_dispersion_ratio), attribute)

	if (nrow(scale_spread_results) == 0) {
		return(invisible(NULL))
	}

	attribute_order <- scale_spread_results$attribute

	plot_data <- df %>%
		filter(paragraph_type_ %in% c("writer", "model")) %>%
		select(paragraph_type_, all_of(scale_attributes)) %>%
		pivot_longer(
			cols = all_of(scale_attributes),
			names_to = "attribute",
			values_to = "rating"
		) %>%
		filter(!is.na(rating), attribute %in% attribute_order) %>%
		mutate(
			attribute = factor(attribute, levels = rev(attribute_order)),
			paragraph_type_ = factor(paragraph_type_, levels = c("writer", "model"))
		)

	plot_summary <- plot_data %>%
		group_by(attribute, paragraph_type_) %>%
		summarise(
			q1 = quantile(rating, 0.25, na.rm = TRUE),
			median = median(rating, na.rm = TRUE),
			q3 = quantile(rating, 0.75, na.rm = TRUE),
			.groups = "drop"
		)

	if (nrow(plot_summary) == 0) {
		return(invisible(NULL))
	}

	boxplot_figure <- ggplot(plot_summary, aes(y = paragraph_type_)) +
		geom_rect(
			aes(xmin = q1, xmax = q3, ymin = as.numeric(paragraph_type_) - 0.28, ymax = as.numeric(paragraph_type_) + 0.28, fill = paragraph_type_),
			colour = "#4d4d4d",
			linewidth = 0.3
		) +
		geom_segment(
			aes(x = median, xend = median, y = as.numeric(paragraph_type_) - 0.28, yend = as.numeric(paragraph_type_) + 0.28),
			linewidth = 0.5,
			colour = "#1f1f1f"
		) +
		facet_grid(attribute ~ ., scales = "fixed", switch = "y") +
		scale_fill_manual(values = c(writer = "#0070C0", model = "#7030A0"), guide = "none") +
		scale_y_discrete(limits = c("writer", "model")) +
		scale_x_continuous(limits = c(0, 100), breaks = c(0, 25, 50, 75, 100)) +
		labs(
			title = paste("Scale rating distributions by paragraph type:", str_to_title(data_split)),
			x = "Rating",
			y = NULL
		) +
		theme_minimal(base_size = 11) +
		theme(
			panel.grid.major.y = element_blank(),
			panel.grid.minor = element_blank(),
			panel.spacing.y = unit(0.15, "lines"),
			strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold", size = 9),
			strip.background = element_rect(fill = "#f5f5f5", colour = NA),
			axis.text.y = element_text(face = "bold"),
			plot.title = element_text(face = "bold")
		)

	ggsave(
		filename = file.path(figure_dir, "scale_boxplots_by_relative_dispersion_ratio.pdf"),
		plot = boxplot_figure,
		width = 11,
		height = max(10, 0.45 * length(attribute_order) + 1.5),
		dpi = 300
	)
}

# ===== RUN ANALYSIS + SAVE OUTPUTS ----
walk(DATA_SPLITS, function(data_split) {
	message("Running spread analysis for split: ", data_split)

	df_split <- get_split_data(data_split)

	spread_results <- bind_rows(
		map_dfr(scale_attributes, ~ {
			message("Running bootstrap spread comparison for scale attribute: ", .x)
			bootstrap_test_attribute(df_split, .x, attribute_type = "scale", n_boot = n_boot)
		}),
		map_dfr(ordinal_attributes, ~ {
			message("Running bootstrap spread comparison for ordinal attribute: ", .x)
			bootstrap_test_attribute(df_split, .x, attribute_type = "ordinal", n_boot = n_boot)
		}),
		map_dfr(nominal_attributes, ~ {
			message("Running bootstrap spread comparison for nominal attribute: ", .x)
			bootstrap_test_attribute(df_split, .x, attribute_type = "nominal", n_boot = n_boot)
		})
	)

	write_csv(
		spread_results,
		file.path(RESULTS_DIR, data_split, "spread_bootstrap_by_type.csv")
	)

	create_scale_spread_boxplot(df_split, spread_results, data_split)
})
