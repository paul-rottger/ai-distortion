# ===== PACKAGES ----
suppressPackageStartupMessages({
	library(readr)
})

# ===== SETTINGS ----
setwd("~/Documents/Repos/ai-distortion")

preferred_results_dir <- "./results/followup_mitigation_phase_2_distortion/preferred"
detail_output_path <- file.path(preferred_results_dir, "mitigation_significance_details.csv")
alpha <- 0.05
distortion_attributes <- c(
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

# ===== HELPERS ----
get_p_value_column <- function(df) {
	if ("p" %in% names(df)) {
		return("p")
	}

	if ("p_value" %in% names(df)) {
		return("p_value")
	}

	stop("No p-value column found in regression results.")
}

extract_attribute_name <- function(file_path) {
	basename(file_path) |>
		stringr::str_remove("_by_mitigation\\.csv$")
}

get_absolute_interval <- function(lower, upper) {
	if (is.na(lower) || is.na(upper)) {
		return(c(low = NA_real_, high = NA_real_))
	}

	if (lower > upper) {
		tmp <- lower
		lower <- upper
		upper <- tmp
	}

	if (lower <= 0 && upper >= 0) {
		return(c(low = 0, high = max(abs(lower), abs(upper))))
	}

	absolute_bounds <- abs(c(lower, upper))
	c(low = min(absolute_bounds), high = max(absolute_bounds))
}

get_relative_magnitude_interval <- function(numerator_low, numerator_high, denominator_low, denominator_high) {
	numerator_interval <- get_absolute_interval(numerator_low, numerator_high)
	denominator_interval <- get_absolute_interval(denominator_low, denominator_high)

	if (
		any(is.na(c(numerator_interval, denominator_interval))) ||
		denominator_interval[["low"]] == 0
	) {
		return(c(low = NA_real_, high = NA_real_))
	}

	c(
		low = numerator_interval[["low"]] / denominator_interval[["high"]],
		high = numerator_interval[["high"]] / denominator_interval[["low"]]
	)
}

load_mitigation_results <- function(file_path) {
	regression_results <- read_csv(file_path, show_col_types = FALSE)
	p_value_column <- get_p_value_column(regression_results)
	target_terms <- c("mitigation_condition_prompting", "mitigation_condition_reranking")

	regression_results$attribute <- extract_attribute_name(file_path)
	regression_results$mitigation <- stringr::str_remove(regression_results$term, "^mitigation_condition_")
	regression_results$p_value <- regression_results[[p_value_column]]
	regression_results$significant <- !is.na(regression_results$p_value) & regression_results$p_value < alpha
	writer_rows <- regression_results[regression_results$term == "mitigation_condition_writer", , drop = FALSE]

	if (nrow(writer_rows) != 1) {
		stop(paste("Expected exactly one writer row in", file_path))
	}

	writer_ame <- writer_rows$ame[[1]]
	writer_ame_low <- writer_rows$ame_low[[1]]
	writer_ame_high <- writer_rows$ame_high[[1]]
	regression_results$writer_ame <- writer_ame
	regression_results$writer_ame_low <- writer_ame_low
	regression_results$writer_ame_high <- writer_ame_high

	if (is.na(writer_ame) || writer_ame == 0) {
		regression_results$relative_ame_magnitude_vs_writer <- NA_real_
		regression_results$relative_ame_magnitude_vs_writer_low <- NA_real_
		regression_results$relative_ame_magnitude_vs_writer_high <- NA_real_
	} else {
		regression_results$relative_ame_magnitude_vs_writer <- abs(regression_results$ame) / abs(writer_ame)
		relative_intervals <- mapply(
			get_relative_magnitude_interval,
			regression_results$ame_low,
			regression_results$ame_high,
			writer_ame_low,
			writer_ame_high
		)
		regression_results$relative_ame_magnitude_vs_writer_low <- relative_intervals["low", ]
		regression_results$relative_ame_magnitude_vs_writer_high <- relative_intervals["high", ]
	}

	regression_results <- regression_results[regression_results$term %in% target_terms, , drop = FALSE]

	ordered_columns <- c(
		"attribute",
		"mitigation",
		"term",
		"p_value",
		"significant",
		"writer_ame",
		"writer_ame_low",
		"writer_ame_high",
		"relative_ame_magnitude_vs_writer",
		"relative_ame_magnitude_vs_writer_low",
		"relative_ame_magnitude_vs_writer_high"
	)
	remaining_columns <- setdiff(names(regression_results), ordered_columns)

	regression_results[, c(ordered_columns, remaining_columns), drop = FALSE]
}

# ===== LOAD ALL PREFERRED-SPLIT REGRESSIONS ----
result_files <- file.path(
	preferred_results_dir,
	paste0(distortion_attributes, "_by_mitigation.csv")
)

if (length(result_files) == 0) {
	stop("No preferred-split mitigation result files were found.")
}

missing_result_files <- result_files[!file.exists(result_files)]

if (length(missing_result_files) > 0) {
	stop(
		paste(
			"Missing preferred-split mitigation result files:",
			paste(basename(missing_result_files), collapse = ", ")
		)
	)
}

mitigation_results <- purrr::map_dfr(result_files, load_mitigation_results)

significance_details <- mitigation_results |>
	dplyr::arrange(.data$mitigation, .data$p_value, .data$attribute)

write_csv(significance_details, detail_output_path)

cat("\nSignificant outcomes by mitigation:\n")
print(
	significance_details |>
		dplyr::filter(.data$significant) |>
		dplyr::select(dplyr::all_of(c("mitigation", "attribute", "p_value")))
)
