#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 1 ANALYSIS: TEXT HOMOGENISATION
#
# Tests whether AI paragraphs on the same proposition are more similar to each
# other than writer paragraphs on that proposition (cf. Padmakumar & He, 2024).
#
# - Embeds every writer, model, and writer-edited model paragraph with a
#   sentence-transformers model (semantic similarity), and represents it as a
#   TF-IDF vector (lexical similarity).
# - For each proposition and paragraph group (writer, AI, AI by model, edited),
#   computes the mean pairwise cosine similarity between paragraphs.
# - Compares AI vs writer similarity paired by proposition (bootstrap CI over
#   propositions and Wilcoxon signed-rank test), and likewise for edited paragraphs.
# - Exports tables to `results/main_phase_1/text_homogenisation/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from demo_paths import get_results_dir, parse_demo_mode
from text_metrics import sample_writer_propositions

DEMO_MODE = parse_demo_mode()
DATA_PATH = BASE_DIR / "data" / "main_phase_1" / "paragraphs.csv"
RESULTS_DIR = get_results_dir(BASE_DIR, "main_phase_1", "text_homogenisation", demo_mode=DEMO_MODE)

SEED = 42
N_BOOTSTRAP = 2000
DEMO_N_PAIRS = 1000
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

SIMILARITY_MEASURES = {
	"embedding": "Semantic similarity (all-mpnet-base-v2 cosine)",
	"tfidf": "Lexical similarity (TF-IDF cosine)",
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

def embed_paragraphs(texts: list[str]) -> dict[str, np.ndarray]:
	"""Unit-normalised vectors per similarity measure, so dot products are cosine similarities."""
	model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION)
	embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
	tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(texts)
	return {"embedding": embeddings, "tfidf": tfidf}


def mean_pairwise_similarity(vectors) -> float:
	n = vectors.shape[0]
	similarities = vectors @ vectors.T
	similarities = similarities.toarray() if hasattr(similarities, "toarray") else similarities
	return (similarities.sum() - np.trace(similarities)) / (n * (n - 1))


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
	"""Long table of (paragraph index, group): writer, AI overall, AI per model, and edited."""
	labels = {"writer": "writer", "model": "ai", "edited": "edited"}
	groups = [df["paragraph_type"].map(labels).rename("group")]
	model_rows = df.loc[df["paragraph_type"] == "model", "model_name"]
	groups.append(("ai: " + model_rows).rename("group"))
	return pd.concat(groups).rename_axis("row").reset_index()


def compute_proposition_similarity(df: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
	groups = assign_groups(df).merge(df[["proposition_id"]], left_on="row", right_index=True)
	rows = []
	for (proposition_id, group), sub in groups.groupby(["proposition_id", "group"]):
		if len(sub) < 2:
			continue
		row = {"proposition_id": proposition_id, "group": group, "n_paragraphs": len(sub)}
		for measure, matrix in vectors.items():
			row[measure] = mean_pairwise_similarity(matrix[sub["row"].to_numpy()])
		rows.append(row)
	return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
	draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
	lower, upper = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
	return values.mean(), lower, upper


def build_summary(similarity: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
	rows = []
	for group, sub in similarity.groupby("group", sort=False):
		for measure, label in SIMILARITY_MEASURES.items():
			mean, lower, upper = bootstrap_mean_ci(sub[measure].to_numpy(), rng)
			rows.append({
				"group": group,
				"measure": measure,
				"label": label,
				"n_propositions": len(sub),
				"mean_paragraphs_per_proposition": sub["n_paragraphs"].mean(),
				"mean": mean,
				"ci_lower": lower,
				"ci_upper": upper,
			})
	order = {"writer": 0, "ai": 1, "edited": 3}
	return pd.DataFrame(rows).sort_values(
		["measure", "group"], key=lambda s: s.map(lambda g: order.get(g, 2)) if s.name == "group" else s
	).reset_index(drop=True)


def build_paired_comparison(similarity: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
	"""Similarity of each AI and edited group minus writer similarity, paired by proposition."""
	wide = similarity.pivot(index="proposition_id", columns="group")
	rows = []
	for group in [g for g in similarity["group"].unique() if g != "writer"]:
		for measure, label in SIMILARITY_MEASURES.items():
			pair = wide[[(measure, "writer"), (measure, group)]].dropna()
			differences = (pair[(measure, group)] - pair[(measure, "writer")]).to_numpy()
			mean, lower, upper = bootstrap_mean_ci(differences, rng)
			statistic, p_value = wilcoxon(differences)
			rows.append({
				"comparison": f"{group} - writer",
				"measure": measure,
				"label": label,
				"n_propositions": len(pair),
				"writer_mean": pair[(measure, "writer")].mean(),
				"comparison_mean": pair[(measure, group)].mean(),
				"mean_difference": mean,
				"ci_lower": lower,
				"ci_upper": upper,
				"pct_propositions_more_similar_than_writer": 100 * (differences > 0).mean(),
				"wilcoxon_statistic": statistic,
				"wilcoxon_p_value": p_value,
			})
	return pd.DataFrame(rows).sort_values(["measure", "comparison"]).reset_index(drop=True)


# =============================================================================
# OUTPUTS
# =============================================================================

def write_outputs(similarity: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	similarity.to_csv(RESULTS_DIR / "similarity_by_proposition.csv", index=False)
	summary.to_csv(RESULTS_DIR / "similarity_summary.csv", index=False)
	paired.to_csv(RESULTS_DIR / "paired_ai_vs_writer.csv", index=False)

	with pd.option_context("display.width", 250, "display.max_columns", None):
		print(summary.drop(columns="label").round(3).to_string(index=False))
		print(paired.drop(columns="label").round(4).to_string(index=False))


def main() -> None:
	rng = np.random.default_rng(SEED)
	df = load_paragraphs()
	vectors = embed_paragraphs(df["paragraph"].tolist())
	similarity = compute_proposition_similarity(df, vectors)
	summary = build_summary(similarity, rng)
	paired = build_paired_comparison(similarity, rng)
	write_outputs(similarity, summary, paired)


if __name__ == "__main__":
	main()
