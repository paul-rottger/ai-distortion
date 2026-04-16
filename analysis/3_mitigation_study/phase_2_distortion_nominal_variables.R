# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
	library(mclogit)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

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
																				outcome_ref = NULL) {
	model_df <- df %>%
		filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
		mutate(
			rater_id = as.factor(rater_id)
		)

	if (nrow(model_df) == 0 || nlevels(model_df[[predictor]]) < 2) {
		return(NULL)
	}

	model_df[[predictor]] <- droplevels(as.factor(model_df[[predictor]]))
	model_df[[predictor]] <- relevel_if_present(model_df[[predictor]], ref = mitigation_reference_level)

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
																	outcome_ref = NULL) {
	model_fit <- fit_multinomial_logit_model(
		df = df,
		outcome = outcome,
		predictor = predictor,
		random_effects = random_effects,
		outcome_ref = outcome_ref
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

# Debug run
debug_sample_size <- min(1000, nrow(data_preferred))
debug_data <- data_preferred %>% slice_sample(n = debug_sample_size)
debug_results <- fit_multinomial_logit(
	debug_data,
	outcome = "writer_gender",
	predictor = "model_and_mitigation_",
	outcome_ref = reference_levels[["writer_gender"]]
)
print(debug_results)

run_nominal_regressions <- function(attribute) {
	print(paste("running multinomial logistic regression for:", attribute))

	for (data_split in c("preferred", "edited", "unedited")) {
		split_data <- switch(data_split,
			unedited = data_unedited,
			edited = data_edited,
			preferred = data_preferred
		)

		dir.create(
			paste0("./results/followup_mitigation_phase_2_distortion/", data_split),
			recursive = TRUE,
			showWarnings = FALSE
		)

		results <- fit_multinomial_logit(
			split_data,
			outcome = attribute,
			predictor = "model_and_mitigation_",
			outcome_ref = reference_levels[[attribute]]
		)

		write_csv(
			results,
			paste0("./results/followup_mitigation_phase_2_distortion/", data_split, "/", attribute, "_by_model_and_mitigation.csv")
		)
	}
}

requested_attributes <- commandArgs(trailingOnly = TRUE)

if (length(requested_attributes) > 0) {
	nominal_vars <- intersect(nominal_vars, requested_attributes)

	if (length(nominal_vars) == 0) {
		stop("No requested nominal variables matched the supported outcomes.")
	}
}

for (attribute in nominal_vars) {
	run_nominal_regressions(attribute)
}
