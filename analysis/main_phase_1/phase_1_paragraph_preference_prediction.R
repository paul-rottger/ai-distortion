suppressPackageStartupMessages({
	library(tidyverse)
	library(broom)
})

# ===== RANDOM SEED ----
set.seed(123)

# ===== DATA IMPORTS ----
setwd("~/Documents/Repos/ai-distortion")

preference_data <- read_csv(
	"./data/main_phase_1/proposition_responses.csv",
	show_col_types = FALSE
)

aggregated_annotations <- read_csv(
	"./data/main_phase_2/annotations_aggregated.csv",
	show_col_types = FALSE
)

# ===== DATA PROCESSING ----
preference_data <- preference_data %>%
	mutate(
		weak_preference_model = as.integer(writer_preference != "original"),
		strict_preference_model = as.integer(writer_preference == "edited"),
		made_edits = as.integer(made_edits)
	)

writer_annotations <- aggregated_annotations %>%
	filter(paragraph_type == "writer") %>%
	transmute(
		writer_id,
		proposition_id,
		writer_stance_polarity_writer = writer_stance_polarity
	)

model_annotations <- aggregated_annotations %>%
	filter(paragraph_type == "model") %>%
	transmute(
		writer_id,
		proposition_id,
		model_name,
		model_input_condition,
		writer_stance_polarity_model = writer_stance_polarity
	)

edited_annotations <- aggregated_annotations %>%
	filter(paragraph_type == "edited") %>%
	transmute(
		writer_id,
		proposition_id,
		model_name,
		model_input_condition,
		writer_stance_polarity_edited = writer_stance_polarity
	)

analysis_data <- preference_data %>%
	left_join(writer_annotations, by = c("writer_id", "proposition_id")) %>%
	left_join(
		model_annotations,
		by = c("writer_id", "proposition_id", "model_name", "model_input_condition")
	) %>%
	left_join(
		edited_annotations,
		by = c("writer_id", "proposition_id", "model_name", "model_input_condition")
	) %>%
	mutate(
		comparison_paragraph_type = case_when(
			!is.na(writer_stance_polarity_edited) ~ "edited",
			!is.na(writer_stance_polarity_model) ~ "model",
			TRUE ~ NA_character_
		),
		comparison_writer_stance_polarity = coalesce(
			writer_stance_polarity_edited,
			writer_stance_polarity_model
		),
		stance_polarity_diff = comparison_writer_stance_polarity - writer_stance_polarity_writer
	) %>%
	filter(
		!is.na(writer_stance_polarity_writer),
		!is.na(comparison_writer_stance_polarity),
		!is.na(stance_polarity_diff)
	) %>%
	select(
		writer_id,
		proposition_id,
		model_name,
		model_input_condition,
		writer_preference,
		made_edits,
		weak_preference_model,
		strict_preference_model,
		comparison_paragraph_type,
		writer_stance_polarity_writer,
		comparison_writer_stance_polarity,
		stance_polarity_diff
	)

dir.create("./results/main_phase_1", recursive = TRUE, showWarnings = FALSE)

write_csv(
	analysis_data,
	"./results/main_phase_1/phase_1_paragraph_preference_prediction_writer_stance_polarity_data.csv"
)

build_model_results <- function(data, outcome) {
	model <- glm(
		formula = reformulate("stance_polarity_diff", response = outcome),
		data = data,
		family = binomial(link = "logit")
	)

	confidence_intervals <- confint.default(model)

	coefficients <- tidy(model) %>%
		mutate(
			outcome = outcome,
			conf.low = confidence_intervals[, 1],
			conf.high = confidence_intervals[, 2],
			odds_ratio = exp(estimate),
			odds_ratio_conf.low = exp(conf.low),
			odds_ratio_conf.high = exp(conf.high)
		) %>%
		select(
			outcome,
			term,
			estimate,
			std.error,
			statistic,
			p.value,
			conf.low,
			conf.high,
			odds_ratio,
			odds_ratio_conf.low,
			odds_ratio_conf.high
		)

	fit_summary <- tibble(
		outcome = outcome,
		n = nobs(model),
		outcome_mean = mean(data[[outcome]]),
		aic = AIC(model),
		bic = BIC(model),
		log_likelihood = as.numeric(logLik(model))
	)

	list(
		model = model,
		coefficients = coefficients,
		fit_summary = fit_summary
	)
}

outcomes <- c("weak_preference_model", "strict_preference_model")

model_results <- map(outcomes, ~ build_model_results(analysis_data, .x))
names(model_results) <- outcomes

coefficient_table <- bind_rows(map(model_results, "coefficients"))
fit_summary_table <- bind_rows(map(model_results, "fit_summary"))

write_csv(
	coefficient_table,
	"./results/main_phase_1/phase_1_paragraph_preference_prediction_writer_stance_polarity_coefficients.csv"
)

write_csv(
	fit_summary_table,
	"./results/main_phase_1/phase_1_paragraph_preference_prediction_writer_stance_polarity_fit_summary.csv"
)

cat("Analysis dataset summary:\n")
print(
	analysis_data %>%
		count(comparison_paragraph_type, made_edits)
)

cat("\nCoefficient table:\n")
print(coefficient_table)

cat("\nFit summary table:\n")
print(fit_summary_table)
