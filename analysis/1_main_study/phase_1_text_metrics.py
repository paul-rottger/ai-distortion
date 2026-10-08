#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 1 ANALYSIS: TEXT METRICS
#
# Compares basic linguistic properties of writer and AI paragraphs.
#
# - Runs LanguageTool (British English, with US spellings also accepted) on
#   every writer, model, and writer-edited model paragraph, and counts errors
#   per bucket (spelling, grammar, punctuation, casing, whitespace,
#   style/other). The headline error rate per 100 words excludes whitespace
#   and style issues.
# - Computes style features: length and readability (Flesch-Kincaid), lexical
#   diversity (MTLD), and first-person pronouns.
# - Summarizes all features by paragraph type, model, and model input
#   condition, with 95% bootstrap CIs clustered by writer.
# - Compares each writer paragraph with the model paragraph for the same
#   writer and proposition (clustered bootstrap CI and Wilcoxon signed-rank test).
# - Identifies words distinctive of writer vs model paragraphs (log-odds ratio
#   with informative Dirichlet prior; Monroe et al., 2008).
# - Exports tables to `results/main_phase_1/text_metrics/` and an error-rate
#   figure to `figures/main_phase_1/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path
import sys
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from demo_paths import get_figures_dir, get_results_dir, parse_demo_mode
from plotting_utils import get_group_offsets
from text_metrics import (
	distinctive_words,
	languagetool_errors,
	languagetool_matches,
	sample_writer_propositions,
	style_features,
)

DEMO_MODE = parse_demo_mode()
DATA_PATH = BASE_DIR / "data" / "main_phase_1" / "paragraphs.csv"
FIGURES_DIR = get_figures_dir(BASE_DIR, "main_phase_1", demo_mode=DEMO_MODE)
RESULTS_DIR = get_results_dir(BASE_DIR, "main_phase_1", "text_metrics", demo_mode=DEMO_MODE)

SEED = 42
N_BOOTSTRAP = 2000
DEMO_N_PAIRS = 300
N_DISTINCTIVE_WORDS = 25

ID_COLUMNS = ["writer_id", "proposition_id", "paragraph_type", "model_name", "model_input_condition"]
PARAGRAPH_TYPES = ["writer", "model", "edited"]
PARAGRAPH_TYPE_LABELS = {"writer": "Writer", "model": "AI", "edited": "AI, edited by writer"}
PARAGRAPH_TYPE_COLORS = {"writer": "#d62728", "model": "#1f77b4", "edited": "#9467bd"}

# Feature column -> (feature group, label), in table order
FEATURES = {
	"errors_per_100_words_headline": ("Errors", "All errors per 100 words (excl. whitespace, style)"),
	"pct_with_any_error": ("Errors", "% paragraphs with at least one error"),
	"errors_per_100_words_spelling": ("Errors", "Spelling errors per 100 words"),
	"errors_per_100_words_grammar": ("Errors", "Grammar errors per 100 words"),
	"errors_per_100_words_punctuation": ("Errors", "Punctuation errors per 100 words"),
	"errors_per_100_words_casing": ("Errors", "Casing errors per 100 words"),
	"errors_per_100_words_whitespace": ("Errors", "Whitespace issues per 100 words"),
	"errors_per_100_words_style_other": ("Errors", "Style / other issues per 100 words"),
	"n_words": ("Length and readability", "Words"),
	"n_sentences": ("Length and readability", "Sentences"),
	"mean_sentence_length": ("Length and readability", "Words per sentence"),
	"mean_word_length": ("Length and readability", "Characters per word"),
	"flesch_kincaid_grade": ("Length and readability", "Flesch-Kincaid grade level"),
	"mtld": ("Lexical diversity", "MTLD"),
	"first_person_singular_per_100_words": ("Personal voice", "First-person singular pronouns per 100 words"),
	"first_person_plural_per_100_words": ("Personal voice", "First-person plural pronouns per 100 words"),
}

ERROR_PLOT_FEATURES = {
	"errors_per_100_words_headline": "All errors\n(excl. whitespace, style)",
	"errors_per_100_words_spelling": "Spelling",
	"errors_per_100_words_grammar": "Grammar",
	"errors_per_100_words_punctuation": "Punctuation",
	"errors_per_100_words_casing": "Casing",
	"errors_per_100_words_whitespace": "Whitespace",
	"errors_per_100_words_style_other": "Style / other",
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_paragraphs() -> pd.DataFrame:
	df = pd.read_csv(DATA_PATH)
	df = df.loc[df["paragraph"].notna() & (df["paragraph"].str.strip() != "")].copy()

	if DEMO_MODE:
		df = sample_writer_propositions(df, DEMO_N_PAIRS, SEED)

	return df.reset_index(drop=True)


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_paragraph_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	texts = df["paragraph"].tolist()
	matches = languagetool_matches(texts)
	errors = languagetool_errors(texts, matches=matches)
	metrics = pd.concat([df[ID_COLUMNS], style_features(texts), errors], axis=1)

	matches = pd.concat(
		[df.loc[matches["text_index"], ID_COLUMNS].reset_index(drop=True), matches.drop(columns="text_index")],
		axis=1,
	)
	return metrics, matches


def cluster_bootstrap_mean_ci(
	values: pd.Series,
	clusters: pd.Series,
	rng: np.random.Generator,
) -> tuple[float, float, float]:
	"""Mean of `values` with a 95% percentile CI from resampling whole clusters (missing values dropped)."""
	data = pd.DataFrame({"value": values.to_numpy(), "cluster": clusters.to_numpy()}).dropna()
	grouped = data.groupby("cluster")["value"]
	sums = grouped.sum().to_numpy()
	counts = grouped.size().to_numpy()
	draws = rng.integers(0, len(sums), size=(N_BOOTSTRAP, len(sums)))
	boot_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
	lower, upper = np.percentile(boot_means, [2.5, 97.5])
	return data["value"].mean(), lower, upper


def summarize_group(group: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
	rows = []
	for feature, (feature_group, label) in FEATURES.items():
		mean, lower, upper = cluster_bootstrap_mean_ci(group[feature].astype(float), group["writer_id"], rng)
		rows.append({
			"feature_group": feature_group,
			"feature": feature,
			"label": label,
			"n_paragraphs": int(group[feature].notna().sum()),
			"n_writers": group["writer_id"].nunique(),
			"mean": mean,
			"ci_lower": lower,
			"ci_upper": upper,
		})
	return rows


def build_summary(metrics: pd.DataFrame, group_col: str, rng: np.random.Generator) -> pd.DataFrame:
	rows = []
	for key, group in metrics.groupby(group_col, sort=False):
		rows.extend({group_col: key, **row} for row in summarize_group(group, rng))
	return pd.DataFrame(rows)


def build_model_breakdown(metrics: pd.DataFrame, group_col: str, rng: np.random.Generator) -> pd.DataFrame:
	"""Model paragraphs by `group_col`, with writer paragraphs as the reference rows."""
	reference = metrics.loc[metrics["paragraph_type"] == "writer"].assign(**{group_col: "writer (reference)"})
	model = metrics.loc[metrics["paragraph_type"] == "model"].sort_values(group_col)
	return build_summary(pd.concat([reference, model]), group_col, rng)


def build_paired_comparison(metrics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
	"""Difference (model minus writer) for paragraphs by the same writer on the same proposition."""
	keys = ["writer_id", "proposition_id"]
	writer = metrics.loc[metrics["paragraph_type"] == "writer"]
	model = metrics.loc[metrics["paragraph_type"] == "model"]
	paired = writer.merge(model, on=keys, suffixes=("_writer", "_model"))

	rows = []
	for feature, (feature_group, label) in FEATURES.items():
		pair_values = paired[[f"{feature}_writer", f"{feature}_model", "writer_id"]].dropna()
		writer_values = pair_values[f"{feature}_writer"].astype(float)
		model_values = pair_values[f"{feature}_model"].astype(float)
		differences = model_values - writer_values
		mean, lower, upper = cluster_bootstrap_mean_ci(differences, pair_values["writer_id"], rng)
		nonzero = differences != 0
		statistic, p_value = (
			wilcoxon(model_values[nonzero], writer_values[nonzero]) if nonzero.any() else (np.nan, np.nan)
		)
		rows.append({
			"feature_group": feature_group,
			"feature": feature,
			"label": label,
			"n_pairs": len(pair_values),
			"n_writers": pair_values["writer_id"].nunique(),
			"writer_mean": writer_values.mean(),
			"model_mean": model_values.mean(),
			"mean_difference": mean,
			"ci_lower": lower,
			"ci_upper": upper,
			"pct_pairs_model_lower": 100 * (differences < 0).mean(),
			"pct_pairs_tied": 100 * (differences == 0).mean(),
			"pct_pairs_model_higher": 100 * (differences > 0).mean(),
			"wilcoxon_statistic": statistic,
			"wilcoxon_p_value": p_value,
		})
	return pd.DataFrame(rows)


def format_estimate(mean: float, lower: float, upper: float) -> str:
	digits = 1 if max(abs(mean), abs(lower), abs(upper)) >= 10 else 2
	return f"{mean:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def format_p_value(p_value: float) -> str:
	if np.isnan(p_value):
		return ""
	return "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"


def build_supplement_table(summary_by_type: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
	"""One row per feature: means [95% CI] by paragraph type, plus the paired writer-vs-model test."""
	table = paired[["feature_group", "feature", "label"]].copy()
	for paragraph_type in PARAGRAPH_TYPES:
		sub = summary_by_type.loc[summary_by_type["paragraph_type"] == paragraph_type].set_index("feature")
		if sub.empty:
			continue
		table[PARAGRAPH_TYPE_LABELS[paragraph_type]] = [
			format_estimate(*sub.loc[feature, ["mean", "ci_lower", "ci_upper"]]) for feature in table["feature"]
		]
	table["AI - writer (paired)"] = [
		format_estimate(row.mean_difference, row.ci_lower, row.ci_upper) for row in paired.itertuples()
	]
	table["p (Wilcoxon)"] = paired["wilcoxon_p_value"].map(format_p_value)
	return table


def build_distinctive_words(metrics_df: pd.DataFrame) -> pd.DataFrame:
	writer_texts = metrics_df.loc[metrics_df["paragraph_type"] == "writer", "paragraph"].tolist()
	model_texts = metrics_df.loc[metrics_df["paragraph_type"] == "model", "paragraph"].tolist()
	return distinctive_words(model_texts, writer_texts).rename(columns={
		"count_a": "count_model",
		"count_b": "count_writer",
		"rate_a_per_10k": "model_per_10k_words",
		"rate_b_per_10k": "writer_per_10k_words",
	})


def build_distinctive_words_table(words: pd.DataFrame) -> pd.DataFrame:
	"""Top words over-represented in AI and in writer paragraphs, side by side."""
	columns = ["word", "model_per_10k_words", "writer_per_10k_words", "z_score"]
	top_model = words.head(N_DISTINCTIVE_WORDS)[columns].reset_index(drop=True)
	top_writer = words.tail(N_DISTINCTIVE_WORDS)[columns].iloc[::-1].reset_index(drop=True)
	return pd.concat(
		[top_model.add_prefix("ai_distinctive_"), top_writer.add_prefix("writer_distinctive_")], axis=1
	).round(2)


# =============================================================================
# OUTPUTS
# =============================================================================

def plot_error_rates(summary: pd.DataFrame, output_path: Path) -> None:
	rates = summary.loc[summary["feature"].isin(ERROR_PLOT_FEATURES)]
	y_positions = {feature: index for index, feature in enumerate(reversed(list(ERROR_PLOT_FEATURES)))}
	offsets = get_group_offsets(list(reversed(PARAGRAPH_TYPES)), offset_scale=0.22)

	fig, ax = plt.subplots(figsize=(6.5, 4.2))
	for paragraph_type in PARAGRAPH_TYPES:
		sub = rates.loc[rates["paragraph_type"] == paragraph_type]
		if sub.empty:
			continue
		y = sub["feature"].map(y_positions) + offsets[paragraph_type]
		ax.errorbar(
			sub["mean"], y,
			xerr=[sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]],
			fmt="o", color=PARAGRAPH_TYPE_COLORS[paragraph_type], capsize=2, markersize=5,
			label=f"{PARAGRAPH_TYPE_LABELS[paragraph_type]} (n={sub['n_paragraphs'].iloc[0]})",
		)

	ax.axhline(len(ERROR_PLOT_FEATURES) - 1.5, color="#808080", linewidth=0.8, linestyle=":")
	ax.set_yticks(list(y_positions.values()))
	ax.set_yticklabels([ERROR_PLOT_FEATURES[feature] for feature in y_positions])
	ax.set_xlabel("LanguageTool errors per 100 words (mean, 95% CI)")
	ax.set_xlim(left=0)
	ax.grid(axis="x", alpha=0.3)
	ax.spines[["top", "right"]].set_visible(False)
	ax.legend(loc="lower right", frameon=False, fontsize=8)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def write_outputs(df: pd.DataFrame, metrics: pd.DataFrame, matches: pd.DataFrame, rng: np.random.Generator) -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)

	metrics.to_csv(RESULTS_DIR / "paragraph_text_metrics.csv", index=False)
	matches.to_csv(RESULTS_DIR / "languagetool_matches.csv", index=False)
	(
		matches.groupby(["paragraph_type", "bucket", "category", "rule_id"]).size()
		.rename("n_matches").reset_index()
		.sort_values(["paragraph_type", "n_matches"], ascending=[True, False])
		.to_csv(RESULTS_DIR / "languagetool_rule_counts.csv", index=False)
	)

	summary_by_type = build_summary(metrics, "paragraph_type", rng)
	summary_by_type.to_csv(RESULTS_DIR / "summary_by_paragraph_type.csv", index=False)
	build_model_breakdown(metrics, "model_name", rng).to_csv(RESULTS_DIR / "summary_by_model.csv", index=False)
	build_model_breakdown(metrics, "model_input_condition", rng).to_csv(
		RESULTS_DIR / "summary_by_input_condition.csv", index=False
	)
	paired = build_paired_comparison(metrics, rng)
	paired.to_csv(RESULTS_DIR / "paired_writer_vs_model.csv", index=False)

	supplement_table = build_supplement_table(summary_by_type, paired)
	supplement_table.to_csv(RESULTS_DIR / "supplement_table_text_features.csv", index=False)

	words = build_distinctive_words(df)
	words.to_csv(RESULTS_DIR / "distinctive_words_writer_vs_model.csv", index=False)
	words_table = build_distinctive_words_table(words)
	words_table.to_csv(RESULTS_DIR / "supplement_table_distinctive_words.csv", index=False)

	plot_error_rates(summary_by_type, FIGURES_DIR / "text_metrics_errors_per_100_words.pdf")

	with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
		print(supplement_table.drop(columns=["feature_group", "feature"]).to_string(index=False))
		print(words_table.to_string(index=False))


def main() -> None:
	rng = np.random.default_rng(SEED)
	df = load_paragraphs()
	metrics, matches = compute_paragraph_metrics(df)
	write_outputs(df, metrics, matches, rng)


if __name__ == "__main__":
	main()
