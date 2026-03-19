# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
})

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORT ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/main_phase_2/annotations.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
	mutate(
		paragraph_type_ = factor(paragraph_type),
		writer_age_binned = factor(writer_age_binned, levels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"), ordered = TRUE),
		writer_english_first = factor(writer_english_first, levels = c("No", "Yes"), ordered = TRUE),
		writer_english_skills = factor(writer_english_skills, levels = c("Basic", "Intermediate", "Advanced", "Expert"), ordered = TRUE),
		writer_education = factor(writer_education, levels = c("GCSEs or equivalent", "A-levels or equivalent", "Vocational qualification", "Undergraduate degree", "Postgraduate degree (Master's)", "Doctorate (PhD)", "Other"), ordered = TRUE),
		writer_income = factor(writer_income, levels = c("Under £15,000", "£15,000-£24,999", "£25,000-£34,999", "£35,000-£49,999", "£50,000-£74,999", "£75,000-£99,999", "£100,000+"), ordered = TRUE)
	)

# Edited dataset for writer vs model comparisons (edited grouped into model)
data_edited <- data %>%
	group_by(writer_id, proposition_id) %>%
	filter(!(paragraph_type == "model" & any(paragraph_type == "edited"))) %>%
	mutate(
		paragraph_type = if_else(paragraph_type == "edited", "model", paragraph_type),
		paragraph_type_ = factor(paragraph_type)
	) %>%
	ungroup() %>%
	filter(paragraph_type_ %in% c("writer", "model")) %>%
	mutate(paragraph_type_ = relevel(paragraph_type_, ref = "writer"))

rm(data)

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

# ===== RUN ANALYSIS + SAVE SINGLE OUTPUT ----
spread_results <- bind_rows(
	map_dfr(scale_attributes, ~ {
		message("Running bootstrap spread comparison for scale attribute: ", .x)
		bootstrap_test_attribute(data_edited, .x, attribute_type = "scale", n_boot = n_boot)
	}),
	map_dfr(ordinal_attributes, ~ {
		message("Running bootstrap spread comparison for ordinal attribute: ", .x)
		bootstrap_test_attribute(data_edited, .x, attribute_type = "ordinal", n_boot = n_boot)
	}),
	map_dfr(nominal_attributes, ~ {
		message("Running bootstrap spread comparison for nominal attribute: ", .x)
		bootstrap_test_attribute(data_edited, .x, attribute_type = "nominal", n_boot = n_boot)
	})
)

write_csv(
	spread_results,
	"./results/main_phase_2_distribution/edited/spread_bootstrap_by_type.csv"
)
