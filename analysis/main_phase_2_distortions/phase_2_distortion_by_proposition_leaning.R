# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(tidyverse)
	library(glmmTMB)
	library(broom.mixed)
	library(marginaleffects)
	library(ordinal)
	library(mclogit)
	library(lme4)
})

source("./analysis/utils_r/variable_definitions.R")
source("./analysis/utils_r/data_loading.R")

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS AND PROCESSING ----
list2env(load_phase2_splits(
	"./data/main_phase_2/annotations.csv",
	"./data/main_phase_1/proposition_responses.csv",
	extra_mutate = function(data) {
		data %>%
			mutate(
				model_           = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_name)), ref = "writer"),
				input_condition_ = relevel(factor(ifelse(paragraph_type == "writer", "writer", model_input_condition)), ref = "writer"),
				across(all_of(ordinal_vars), ~ factor(.x, levels = ordinal_levels[[cur_column()]], ordered = TRUE))
			)
	}
), envir = environment())

# Join proposition leaning onto preferred split and restrict to left/right
propositions <- read_csv("./data/main_phase_1/propositions.csv", show_col_types = FALSE)
data_preferred <- data_preferred %>%
	left_join(
		propositions %>% select(proposition_id, proposition_leaning),
		by = "proposition_id"
	) %>%
	filter(proposition_leaning %in% c("left", "right"))
rm(propositions)

preferred_leaning_counts <- data_preferred %>%
	count(proposition_leaning, name = "n_observations") %>%
	complete(
		proposition_leaning = c("left", "right"),
		fill = list(n_observations = 0)
	)

print("preferred subset observation counts by proposition leaning:")
preferred_leaning_counts %>%
	arrange(match(proposition_leaning, c("left", "right"))) %>%
	pwalk(function(proposition_leaning, n_observations) {
		print(paste(proposition_leaning, "=", n_observations))
	})

output_dir <- "./results/main_phase_2_distortion/preferred/by_proposition_leaning"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ===== BETA REGRESSION SETUP for SCALE VARIABLES ----

squeeze01 <- function(y) {
	n <- length(y)
	(y * (n - 1) + 0.5) / n
}

fit_beta <- function(df, outcome, predictor, random) {
	y <- df[[outcome]] / 100
	y <- pmin(pmax(y, 0), 1)
	y <- squeeze01(y)

	form <- as.formula(paste0("y ~ ", predictor, " + ", random))
	model <- glmmTMB(form, data = df, family = beta_family(link = "logit"))

	tidy_fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
		filter(term != "(Intercept)") %>%
		mutate(
			odds_ratio = exp(estimate),
			or_low = exp(conf.low),
			or_high = exp(conf.high),
			p = p.value
		) %>%
		select(term, odds_ratio, or_low, or_high, statistic, p)

	ame <- avg_comparisons(
		model,
		variables = setNames(list("reference"), predictor),
		type = "response",
		re.form = NA
	) %>%
		as_tibble() %>%
		mutate(
			term = paste0(predictor, sub(" .*", "", contrast)),
			ame = estimate * 100,
			ame_low = conf.low * 100,
			ame_high = conf.high * 100
		) %>%
		select(term, ame, ame_low, ame_high)

	tidy_fixed <- tidy_fixed %>% left_join(ame, by = "term")

	list(model = model, tidy_fixed = tidy_fixed)
}

# ===== ORDINAL LOGISTIC REGRESSION SETUP ----

fit_ordinal_logit <- function(df, outcome, predictor = "paragraph_type_", random = "(1 | rater_id)") {
	model_df <- df %>%
		filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
		mutate(rater_id = as.factor(rater_id))

	if (predictor == "paragraph_type_") {
		model_df <- model_df %>%
			mutate(paragraph_type_ = relevel(as.factor(paragraph_type_), ref = "writer"))
	}

	if (outcome == "writer_education") {
		model_df <- model_df %>%
			filter(writer_education != "Other")
	}

	form <- as.formula(paste0(outcome, " ~ ", predictor, " + ", random))
	model <- clmm(
		form,
		data = model_df,
		link = "logit",
		Hess = TRUE,
		nAGQ = 0,
		control = clmm.control(maxIter = 200, gradTol = 1e-4)
	)

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
		dplyr::select(term, odds_ratio, or_low, or_high, statistic, p, p_value)

	list(model = model, tidy_fixed = tidy_fixed)
}

# ===== NOMINAL LOGISTIC REGRESSION SETUP ----

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
		p_value = numeric()
	)
}

relevel_if_present <- function(x, ref) {
	if (ref %in% levels(x)) {
		return(relevel(x, ref = ref))
	}

	x
}

prepare_predictor <- function(model_df, predictor) {
	model_df[[predictor]] <- droplevels(as.factor(model_df[[predictor]]))
	model_df[[predictor]] <- relevel_if_present(model_df[[predictor]], ref = "writer")
	model_df
}

get_multinomial_random_effects <- function(attribute) {
	if (attribute %in% c("writer_politicalParty", "writer_politicalIdeology")) {
		return(NULL)
	}

	~1 | rater_id
}

fit_multinomial_logit_model <- function(df,
									outcome,
									predictor = "paragraph_type_",
									random_effects = ~1 | rater_id,
									outcome_ref = NULL) {
	model_df <- df %>%
		filter(!is.na(.data[[outcome]]), !is.na(.data[[predictor]])) %>%
		mutate(rater_id = as.factor(rater_id))

	model_df <- prepare_predictor(model_df, predictor)

	if (nrow(model_df) == 0 || nlevels(model_df[[predictor]]) < 2) {
		return(NULL)
	}

	model_df[[outcome]] <- as.factor(model_df[[outcome]])
	model_df[[outcome]] <- droplevels(model_df[[outcome]])

	if (!is.null(outcome_ref) && outcome_ref %in% levels(model_df[[outcome]])) {
		model_df[[outcome]] <- relevel(model_df[[outcome]], ref = outcome_ref)
	}

	if (nlevels(model_df[[outcome]]) < 2) {
		return(NULL)
	}

	form <- as.formula(paste0(outcome, " ~ ", predictor))

	fit_args <- list(
		formula = form,
		data = model_df,
		estimator = "ML"
	)

	if (!is.null(random_effects)) {
		fit_args$random <- random_effects
	}

	fit <- suppressWarnings(do.call(mclogit::mblogit, fit_args))

	if (is.null(fit)) {
		return(NULL)
	}

	list(model = fit, model_df = model_df)
}

fit_multinomial_logit <- function(df,
								outcome,
								predictor = "paragraph_type_",
								random_effects = ~1 | rater_id,
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

	coef_table <- as_tibble(coefs, rownames = "row_id") %>%
		separate(row_id, into = c("target_level", "term"), sep = "~", remove = TRUE) %>%
		filter(str_starts(term, predictor))

	if (nrow(coef_table) == 0) {
		reference_level <- levels(model_fit$model_df[[outcome]])[1]
		predictor_levels <- levels(model_fit$model_df[[predictor]])
		predictor_terms <- paste0(predictor, predictor_levels[predictor_levels != predictor_levels[1]])
		target_levels <- setdiff(levels(model_fit$model_df[[outcome]]), reference_level)

		if (length(predictor_terms) == 0 || length(target_levels) == 0) {
			return(empty_nominal_results())
		}

		return(expand_grid(
			term = predictor_terms,
			target_level = target_levels
		) %>%
			mutate(
				reference_level = reference_level,
				odds_ratio = NA_real_,
				or_low = NA_real_,
				or_high = NA_real_,
				statistic = NA_real_,
				p = NA_real_,
				p_value = NA_real_
			))
	}

	coef_table %>%
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

run_scale_regressions <- function(attribute) {
	print(paste("running beta regression for:", attribute))

	for (leaning in c("left", "right")) {
		split_data <- data_preferred %>%
			filter(proposition_leaning == leaning)

		results <- fit_beta(split_data,
			outcome = attribute,
			predictor = "paragraph_type_",
			random = "(1 | rater_id)"
		)

		write_csv(
			results$tidy_fixed %>% mutate(proposition_leaning = leaning, .before = 1),
			paste0(output_dir, "/", attribute, "_", leaning, ".csv")
		)
	}
}

run_ordinal_regressions <- function(attribute) {
	print(paste("running ordinal logistic regression for:", attribute))

	for (leaning in c("left", "right")) {
		split_data <- data_preferred %>%
			filter(proposition_leaning == leaning)

		results <- fit_ordinal_logit(
			split_data,
			outcome = attribute,
			predictor = "paragraph_type_",
			random = "(1 | rater_id)"
		)

		write_csv(
			results$tidy_fixed %>% mutate(proposition_leaning = leaning, .before = 1),
			paste0(output_dir, "/", attribute, "_", leaning, ".csv")
		)
	}
}

run_nominal_regressions <- function(attribute) {
	print(paste("running multinomial logistic regression for:", attribute))

	multinomial_random_effects <- get_multinomial_random_effects(attribute)

	for (leaning in c("left", "right")) {
		split_data <- data_preferred %>%
			filter(proposition_leaning == leaning)

		multinomial_results <- fit_multinomial_logit(
			split_data,
			predictor = "paragraph_type_",
			outcome = attribute,
			random_effects = multinomial_random_effects,
			outcome_ref = reference_levels[[attribute]]
		)

		write_csv(
			multinomial_results %>% mutate(proposition_leaning = leaning, .before = 1),
			paste0(output_dir, "/", attribute, "_", leaning, ".csv")
		)
	}
}

# ===== RATING ATTRIBUTES ----

scale_attributes <- c(
	"paragraph_formality",
	"paragraph_clarity",
	"paragraph_informativeness",
	"paragraph_originality",
	"paragraph_relevance",
	"writer_knowledge",
	"writer_importance",
	"writer_confidence",
	"writer_stance",
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

nominal_attributes <- c(
	"writer_race",
	"writer_gender",
	"writer_politicalParty",
	"writer_politicalIdeology"
)

ordinal_attributes <- c(
	"writer_english_first",
	"writer_education",
	"writer_english_skills",
	"writer_income",
	"writer_age_binned"
)

for (attr in scale_attributes) {
	run_scale_regressions(attr)
}

for (attr in nominal_attributes) {
	run_nominal_regressions(attr)
}

for (attr in ordinal_attributes) {
	run_ordinal_regressions(attr)
}
