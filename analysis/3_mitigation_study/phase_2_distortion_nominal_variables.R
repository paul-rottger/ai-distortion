#!/usr/bin/env Rscript

# =============================================================================
# FOLLOWUP MITIGATION STUDY - PHASE 2 DISTORTION ANALYSIS: NOMINAL VARIABLES
#
# Estimates nominal distortion patterns across mitigation-conditioned model output.
#
# - Fits multinomial and one-vs-all logistic models for nominal outcomes.
# - Compares model-by-mitigation combinations against writer baselines.
# - Runs analyses on unedited, edited, and preferred subsets.
# - Writes nominal distortion result tables to results/followup_mitigation_phase_2_distortion/.
#
# =============================================================================

# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
	library(mclogit)
})

source("./analysis/utils_r/demo_paths.R")
source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== COMMAND-LINE FLAGS ----
args <- commandArgs(trailingOnly = TRUE)
demo_mode <- parse_demo_mode(args)
RESULTS_DIR <- get_results_dir(demo_mode, "followup_mitigation_phase_2_distortion")

mitigation_reference_level <- "writer"

relevel_if_present <- function(x, ref) {
	if (ref %in% levels(x)) {
		relevel(x, ref = ref)
	} else {
		x
	}
}

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
	"./data/followup_mitigation_phase_2/annotations.csv",
	"./data/followup_mitigation_phase_1/proposition_responses.csv",
	extra_mutate = function(data) {
		data %>%
			mutate(
				model_                = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
				mitigation_condition_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_mitigation_condition)), ref = "writer"),
				model_and_mitigation_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", paste(model_name, model_mitigation_condition, sep = "__"))), ref = "writer")
			)
	}
), envir = environment())

# ===== NOMINAL VARIABLES / REFERENCE CATEGORIES ----
reference_levels <- c(
	writer_race = "White",
	writer_gender = "Male",
	writer_politicalParty = "Labour",
	writer_politicalIdeology = "Centrist"
)

empty_nominal_results <- function() {
	tibble(
		term = character(),
		target_level = character(),
		reference_level = character(),
		odds_ratio = numeric(),
		or_low = numeric(),
		or_high = numeric(),
		statistic = numeric(),
		p = numeric(),
		p_value = numeric(),
	)
}

# ===== MULTINOMIAL LOGISTIC REGRESSION (BY MITIGATION) ----
fit_multinomial_logit_model <- function(df,
																				outcome,
																		predictor = "model_and_mitigation_",
																				random_effects = ~ 1 | rater_id,
																				outcome_ref = NULL,
																				mitigation_ref = mitigation_reference_level) {
	model_df <- df %>%
		filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
		mutate(
			rater_id = as.factor(rater_id)
		)

	if (nrow(model_df) == 0 || nlevels(model_df[[predictor]]) < 2) {
		return(NULL)
	}

	model_df[[predictor]] <- droplevels(as.factor(model_df[[predictor]]))
	model_df[[predictor]] <- relevel_if_present(model_df[[predictor]], ref = mitigation_ref)

	model_df[[outcome]] <- as.factor(model_df[[outcome]])
	model_df[[outcome]] <- droplevels(model_df[[outcome]])

	if (!is.null(outcome_ref) && outcome_ref %in% levels(model_df[[outcome]])) {
		model_df[[outcome]] <- relevel(model_df[[outcome]], ref = outcome_ref)
	}

	if (nlevels(model_df[[outcome]]) < 2) {
		return(NULL)
	}

	form <- as.formula(paste0(outcome, " ~ ", predictor))

	fit <- tryCatch(
		suppressWarnings(
			mclogit::mblogit(
				form,
				random = random_effects,
				data = model_df,
				estimator = "ML"
			)
		),
		error = function(e) NULL
	)

	if (is.null(fit)) {
		return(NULL)
	}

	list(
		model = fit,
		model_df = model_df
	)
}


fit_multinomial_logit <- function(df,
																	outcome,
																predictor = "model_and_mitigation_",
																	random_effects = ~ 1 | rater_id,
																	outcome_ref = NULL,
																	mitigation_ref = mitigation_reference_level) {
	model_fit <- fit_multinomial_logit_model(
		df = df,
		outcome = outcome,
		predictor = predictor,
		random_effects = random_effects,
		outcome_ref = outcome_ref,
		mitigation_ref = mitigation_ref
	)

	if (is.null(model_fit)) {
		return(empty_nominal_results())
	}

	fit_summary <- summary(model_fit$model)
	coefs <- fit_summary$coefficients

	if (is.null(dim(coefs))) {
		coefs <- matrix(coefs, nrow = 1)
		colnames(coefs) <- names(fit_summary$coefficients)
		rownames(coefs) <- paste0(levels(model_fit$model_df[[outcome]])[2], "~(Intercept)")
	}

	term_prefix <- paste0("^", predictor)

	as_tibble(coefs, rownames = "row_id") %>%
		separate(row_id, into = c("target_level", "term"), sep = "~", remove = TRUE) %>%
		filter(str_detect(term, term_prefix)) %>%
		transmute(
			term = term,
			target_level = target_level,
			reference_level = levels(model_fit$model_df[[outcome]])[1],
			odds_ratio = exp(Estimate),
			or_low = exp(Estimate - 1.96 * `Std. Error`),
			or_high = exp(Estimate + 1.96 * `Std. Error`),
			statistic = `z value`,
			p = `Pr(>|z|)`,
			p_value = `Pr(>|z|)`
		)
}

run_nominal_regressions <- function(attribute) {
	print(paste("running multinomial logistic regression for:", attribute))

	for (data_split in if (demo_mode) c("preferred") else c("preferred", "edited", "unedited")) {
		split_data <- switch(data_split,
			unedited = data_unedited,
			edited = data_edited,
			preferred = data_preferred
		)

		if (demo_mode) {
			split_data <- split_data %>%
				slice_sample(n = min(200, nrow(split_data)))
		}

		dir.create(
			file.path(RESULTS_DIR, data_split),
			recursive = TRUE,
			showWarnings = FALSE
		)

		for (predictor in list(
			c("mitigation_condition_", "by_mitigation"),
			c("model_and_mitigation_", "by_model_and_mitigation")
		)) {
			results <- fit_multinomial_logit(
				split_data,
				outcome = attribute,
				predictor = predictor[1],
				outcome_ref = reference_levels[[attribute]]
			)

			write_csv(
				results,
				file.path(RESULTS_DIR, data_split, paste0(attribute, "_", predictor[2], ".csv"))
			)

			if (predictor[2] == "by_mitigation") {
				none_dir <- file.path(RESULTS_DIR, data_split, "ref_none")
				dir.create(none_dir, recursive = TRUE, showWarnings = FALSE)

				results_none <- fit_multinomial_logit(
					split_data,
					outcome = attribute,
					predictor = predictor[1],
					outcome_ref = reference_levels[[attribute]],
					mitigation_ref = "none"
				)
				write_csv(
					results_none,
					file.path(none_dir, paste0(attribute, "_", predictor[2], ".csv"))
				)
			}
		}
	}
}

requested_attributes <- setdiff(args, "demo")

if (length(requested_attributes) > 0) {
	nominal_vars <- intersect(nominal_vars, requested_attributes)

	if (length(nominal_vars) == 0) {
		stop("No requested nominal variables matched the supported outcomes.")
	}
}

if (demo_mode) {
	message("Running in demo mode on n=200 samples from each data split.")
}

for (attribute in nominal_vars) {
	run_nominal_regressions(attribute)
}
