# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
	library(ordinal)
})

source("./analysis/utils_r/variable_definitions.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")
data <- read_csv("./data/followup_mitigation_phase_2/annotations.csv", show_col_types = FALSE)
phase_1_preferences <- read_csv("./data/followup_mitigation_phase_1/proposition_responses.csv", show_col_types = FALSE)

# ===== DATA PROCESSING ----
data <- data %>%
	mutate(
		rater_id = as.factor(rater_id),
		writer_id = as.factor(writer_id),
		proposition_id = as.factor(proposition_id),
		model_ = factor(ifelse(
			paragraph_type == "writer", "writer", model_name
		)),
		input_condition_ = factor(ifelse(
			paragraph_type == "writer", "writer", model_input_condition
		)),
		mitigation_condition_ = factor(ifelse(
			paragraph_type == "writer", "writer", model_mitigation_condition
		)),
		model_and_mitigation_ = factor(ifelse(
			paragraph_type == "writer",
			"writer",
			paste(model_name, model_mitigation_condition, sep = "__")
		)),
		across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
	)

# Set reference category for predictors
data$model_ <- relevel(data$model_, ref = "writer")
data$input_condition_ <- relevel(data$input_condition_, ref = "writer")
data$mitigation_condition_ <- relevel(data$mitigation_condition_, ref = "writer")
data$model_and_mitigation_ <- relevel(data$model_and_mitigation_, ref = "writer")

# Create unedited and edited subsets of data for later analyses
data_unedited <- data %>%
	filter(paragraph_type %in% c("writer", "model")) %>%
	mutate(
		paragraph_type_ = as.factor(paragraph_type),
		paragraph_type_ = relevel(paragraph_type_, ref = "writer")
	)

data_edited <- data %>%
	group_by(writer_id, proposition_id) %>%
	filter(!(paragraph_type == "model" &
		any(paragraph_type == "edited"))) %>%
	mutate(
		paragraph_type = if_else(paragraph_type == "edited",
			"model",
			paragraph_type
		),
		paragraph_type_ = as.factor(paragraph_type),
		paragraph_type_ = relevel(paragraph_type_, ref = "writer")
	) %>%
	ungroup()

preferred_exclusions <- phase_1_preferences %>%
	filter(writer_preference == "original") %>%
	mutate(
		writer_id = as.factor(writer_id),
		proposition_id = as.factor(proposition_id)
	) %>%
	distinct(writer_id, proposition_id)

data_preferred <- data_edited %>%
	anti_join(preferred_exclusions, by = c("writer_id", "proposition_id"))

rm(data, phase_1_preferences, preferred_exclusions)


# ===== ORDINAL LOGISTIC REGRESSION (BY MITIGATION) ----

empty_ordinal_results <- function() {
	tibble(
		term = character(),
		odds_ratio = numeric(),
		or_low = numeric(),
		or_high = numeric(),
		statistic = numeric(),
		p = numeric(),
		p_value = numeric()
	)
}

fit_ordinal_logit <- function(df, outcome, predictor = "model_and_mitigation_", random = "(1 | rater_id)") {
	model_df <- df %>%
		filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
		mutate(
			rater_id = as.factor(rater_id)
		)

	model_df[[predictor]] <- droplevels(as.factor(model_df[[predictor]]))

	if (outcome == "writer_education") {
		model_df <- model_df %>%
			filter(writer_education != "Other") %>%
			mutate(writer_education = droplevels(writer_education))
	}

	if (nrow(model_df) == 0 || nlevels(model_df[[predictor]]) < 2 || nlevels(model_df[[outcome]]) < 2) {
		return(list(model = NULL, tidy_fixed = empty_ordinal_results()))
	}

	form <- as.formula(paste0(outcome, " ~ ", predictor, " + ", random))
	model <- tryCatch(
		clmm(
			form,
			data = model_df,
			link = "logit",
			Hess = TRUE,
			nAGQ = 0,
			control = clmm.control(maxIter = 200, gradTol = 1e-4)
		),
		error = function(e) NULL
	)

	if (is.null(model)) {
		return(list(model = NULL, tidy_fixed = empty_ordinal_results()))
	}

	coef_tbl <- as_tibble(summary(model)$coefficients, rownames = "term")

	tidy_fixed <- coef_tbl %>%
		filter(str_starts(term, predictor)) %>%
		transmute(
			term = term,
			estimate = Estimate,
			std_error = `Std. Error`,
			statistic = `z value`,
			p = `Pr(>|z|)`,
			conf.low = estimate - 1.96 * std_error,
			conf.high = estimate + 1.96 * std_error,
			odds_ratio = exp(estimate),
			or_low = exp(conf.low),
			or_high = exp(conf.high),
			p_value = p
		) %>%
		select(term, odds_ratio, or_low, or_high, statistic, p, p_value)

	list(model = model, tidy_fixed = tidy_fixed)
}

# Debug run
debug_sample_size <- min(1000, nrow(data_preferred))
debug_data <- data_preferred %>% slice_sample(n = debug_sample_size)
debug_results <- fit_ordinal_logit(debug_data,
	outcome = "writer_income",
	predictor = "model_and_mitigation_",
	random = "(1 | rater_id)"
)
print(debug_results$tidy_fixed)

run_ordinal_regressions <- function(attribute) {
	print(paste("running ordinal logistic regression for:", attribute))

	for (data_split in c("preferred")) {
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

		results <- fit_ordinal_logit(
			split_data,
			outcome = attribute,
			predictor = "model_and_mitigation_",
			random = "(1 | rater_id)"
		)

		write_csv(
			results$tidy_fixed,
			paste0("./results/followup_mitigation_phase_2_distortion/", data_split, "/", attribute, "_by_model_and_mitigation.csv")
		)
	}
}

for (attr in ordinal_vars) {
	run_ordinal_regressions(attr)
}
