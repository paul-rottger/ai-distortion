"""Per-paragraph text metrics for comparing human and AI writing.

Each metric function takes a list of texts and returns a DataFrame with one row per text,
so further metrics can be added here and concatenated column-wise.
"""

from __future__ import annotations

import math
import re

import language_tool_python
import numpy as np
import pandas as pd
import pyphen
from tqdm import tqdm

LANGUAGETOOL_VERSION = "6.8"

# Writers are UK-based, so we check against British English. British spelling hits are only
# counted if US English also flags them, so neither dialect's spelling counts as an error.
PRIMARY_LANGUAGE = "en-GB"
SECONDARY_LANGUAGE = "en-US"
DIALECT_SPELLING_RULES = {"MORFOLOGIK_RULE_EN_GB"}

# Dictionary spelling hits on acronym-like tokens (e.g. "UBI", "ISAs") are not counted
ACRONYM_PATTERN = re.compile(r"^(?:[^a-z]*[A-Z]){2}")

# Categories about British/American variant choice rather than errors
DISABLED_CATEGORIES = ["BRE_STYLE_OXFORD_SPELLING", "AMERICAN_ENGLISH"]

ERROR_BUCKETS = ["spelling", "grammar", "punctuation", "casing", "whitespace", "style_other"]
HEADLINE_BUCKETS = ["spelling", "grammar", "punctuation", "casing"]

CATEGORY_TO_BUCKET = {
	"TYPOS": "spelling",
	"GRAMMAR": "grammar",
	"PUNCTUATION": "punctuation",
	"CASING": "casing",
}

# Rules filed under LanguageTool's MISC category that are clear grammar errors
RULE_TO_BUCKET = {"EN_A_VS_AN": "grammar"}

WORD_PATTERN = re.compile(r"\b[\w'’-]+\b")


def sample_writer_propositions(df: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
	"""Keep all paragraph rows for a random sample of writer x proposition pairs (used in demo mode)."""
	pairs = df[["writer_id", "proposition_id"]].drop_duplicates()
	sampled_pairs = pairs.sample(n=min(n_pairs, len(pairs)), random_state=seed)
	return df.merge(sampled_pairs, on=["writer_id", "proposition_id"])


def _escape_line_breaks(text: str) -> str:
	return text.replace("\r", "\\r").replace("\n", "\\n")


def count_words(text: str) -> int:
	return len(WORD_PATTERN.findall(text))


def classify_match(match: language_tool_python.Match) -> str:
	if "WHITESPACE" in match.rule_id or "SPACE" in match.rule_id:
		return "whitespace"
	if match.rule_id in RULE_TO_BUCKET:
		return RULE_TO_BUCKET[match.rule_id]
	return CATEGORY_TO_BUCKET.get(match.category, "style_other")


def _make_tool(language: str) -> language_tool_python.LanguageTool:
	tool = language_tool_python.LanguageTool(language, language_tool_download_version=LANGUAGETOOL_VERSION)
	tool.disabled_categories.update(DISABLED_CATEGORIES)
	return tool


def languagetool_matches(texts: list[str]) -> pd.DataFrame:
	"""Return one row per counted LanguageTool match, with `text_index`, rule, bucket and flagged span."""
	primary = _make_tool(PRIMARY_LANGUAGE)
	secondary = _make_tool(SECONDARY_LANGUAGE)
	rows = []
	try:
		for text_index, text in enumerate(tqdm(texts, desc="LanguageTool")):
			matches = primary.check(text)
			if any(m.rule_id in DIALECT_SPELLING_RULES for m in matches):
				secondary_spans = {
					(m.offset, m.error_length) for m in secondary.check(text) if m.category == "TYPOS"
				}
				matches = [
					m for m in matches
					if m.rule_id not in DIALECT_SPELLING_RULES or (m.offset, m.error_length) in secondary_spans
				]
			matches = [
				m for m in matches
				if m.rule_id not in DIALECT_SPELLING_RULES
				or not ACRONYM_PATTERN.match(text[m.offset:m.offset + m.error_length])
			]
			for m in matches:
				rows.append({
					"text_index": text_index,
					"rule_id": m.rule_id,
					"category": m.category,
					"bucket": classify_match(m),
					"flagged": _escape_line_breaks(text[m.offset:m.offset + m.error_length]),
					"context": _escape_line_breaks(m.context),
				})
	finally:
		primary.close()
		secondary.close()
	return pd.DataFrame(rows, columns=["text_index", "rule_id", "category", "bucket", "flagged", "context"])


def languagetool_errors(texts: list[str], matches: pd.DataFrame | None = None) -> pd.DataFrame:
	"""Count LanguageTool errors per bucket for each text, plus headline errors per 100 words."""
	if matches is None:
		matches = languagetool_matches(texts)
	counts = (
		matches.groupby(["text_index", "bucket"]).size()
		.unstack(fill_value=0)
		.reindex(index=range(len(texts)), columns=ERROR_BUCKETS, fill_value=0)
		.fillna(0)
		.astype(int)
		.add_prefix("errors_")
	)
	n_words = np.array([count_words(text) for text in texts])
	counts["errors_headline"] = counts[[f"errors_{b}" for b in HEADLINE_BUCKETS]].sum(axis=1)
	for bucket in ERROR_BUCKETS + ["headline"]:
		counts[f"errors_per_100_words_{bucket}"] = 100 * counts[f"errors_{bucket}"] / n_words
	counts["pct_with_any_error"] = 100 * (counts["errors_headline"] > 0)
	return counts.reset_index(drop=True)


# =============================================================================
# STYLE FEATURES
# =============================================================================

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"'”’)]*\s+|[\r\n]+")
MTLD_THRESHOLD = 0.72
HYPHENATOR = pyphen.Pyphen(lang="en_GB")

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'd", "we'll"}


def _normalize(text: str) -> str:
	return text.replace("’", "'").replace("‘", "'")


def tokenize(text: str) -> list[str]:
	return WORD_PATTERN.findall(_normalize(text))


def split_sentences(text: str) -> list[str]:
	sentences = [s for s in SENTENCE_BOUNDARY.split(text.strip()) if WORD_PATTERN.search(s)]
	return sentences or [text]


def count_syllables(word: str) -> int:
	return max(1, len(HYPHENATOR.positions(word)) + 1)


def flesch_kincaid_grade(n_words: int, n_sentences: int, n_syllables: int) -> float:
	return 0.39 * n_words / n_sentences + 11.8 * n_syllables / n_words - 15.59


def mtld(tokens: list[str], threshold: float = MTLD_THRESHOLD) -> float:
	"""Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010), averaged over both directions."""

	def one_direction(sequence: list[str]) -> float:
		factors, types, count = 0.0, set(), 0
		for token in sequence:
			count += 1
			types.add(token)
			if len(types) / count <= threshold:
				factors += 1
				types, count = set(), 0
		if count > 0:
			factors += (1 - len(types) / count) / (1 - threshold)
		return len(sequence) / factors if factors > 0 else math.nan

	values = [value for value in (one_direction(tokens), one_direction(tokens[::-1])) if not math.isnan(value)]
	return sum(values) / len(values) if values else math.nan


def style_features(texts: list[str]) -> pd.DataFrame:
	"""Surface, readability, lexical-diversity and personal-voice features for each text."""
	rows = []
	for text in tqdm(texts, desc="Style features"):
		raw_tokens = tokenize(text)
		tokens = [token.lower() for token in raw_tokens]
		n_words = len(tokens)
		per_100 = 100 / n_words
		sentences = split_sentences(text)

		rows.append({
			"n_words": n_words,
			"n_sentences": len(sentences),
			"mean_sentence_length": n_words / len(sentences),
			"mean_word_length": np.mean([len(token) for token in tokens]),
			"flesch_kincaid_grade": flesch_kincaid_grade(
				n_words, len(sentences), sum(count_syllables(token) for token in tokens)
			),
			"mtld": mtld(tokens),
			"first_person_singular_per_100_words": per_100 * sum(t in FIRST_PERSON_SINGULAR for t in tokens),
			# Uppercase "US" is the country, not the pronoun
			"first_person_plural_per_100_words": per_100 * sum(
				t in FIRST_PERSON_PLURAL and raw != "US" for t, raw in zip(tokens, raw_tokens)
			),
		})
	return pd.DataFrame(rows)


# =============================================================================
# DISTINCTIVE VOCABULARY
# =============================================================================

def distinctive_words(texts_a: list[str], texts_b: list[str], min_count: int = 20) -> pd.DataFrame:
	"""Log-odds ratios with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008).

	The prior is the pooled word counts of both corpora. Positive z-scores mean a word is
	over-represented in `texts_a`, negative z-scores that it is over-represented in `texts_b`.
	"""
	counts_a = pd.Series([t.lower() for text in texts_a for t in tokenize(text)]).value_counts()
	counts_b = pd.Series([t.lower() for text in texts_b for t in tokenize(text)]).value_counts()
	counts = pd.concat([counts_a.rename("count_a"), counts_b.rename("count_b")], axis=1).fillna(0)
	prior = counts["count_a"] + counts["count_b"]
	n_a, n_b, prior_total = counts["count_a"].sum(), counts["count_b"].sum(), prior.sum()

	log_odds_a = np.log((counts["count_a"] + prior) / (n_a + prior_total - counts["count_a"] - prior))
	log_odds_b = np.log((counts["count_b"] + prior) / (n_b + prior_total - counts["count_b"] - prior))
	delta = log_odds_a - log_odds_b
	variance = 1 / (counts["count_a"] + prior) + 1 / (counts["count_b"] + prior)

	result = counts.assign(
		rate_a_per_10k=10_000 * counts["count_a"] / n_a,
		rate_b_per_10k=10_000 * counts["count_b"] / n_b,
		log_odds_ratio=delta,
		z_score=delta / np.sqrt(variance),
	)
	result = result.loc[prior >= min_count].rename_axis("word").reset_index()
	return result.sort_values("z_score", ascending=False).reset_index(drop=True)
