#!/usr/bin/env python3

# =============================================================================
# DISCLAIMER STUDY - PHASE 1 ANALYSIS: PREFERENCE REASONS (LLM CODING)
#
# Codes writers' free-text preference reasons with an LLM (OpenRouter) using
# `analysis/utils_files/preference_reason_codebook.json`. Each response gets all
# codes that apply (multi-label).
#
# - Calls the API, so it is NOT part of run_all.sh. API responses are cached in
#   `llm_cache.jsonl`, so re-runs only code responses that are not cached yet.
# - Writes `results/followup_disclaimer_phase_1/preference_reasons/reason_codes.csv`,
#   which is summarised by `phase_1_preference_reasons_summary.py`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import argparse
from pathlib import Path
import sys

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from demo_paths import get_results_dir
from preference_reasons import DEFAULT_MODEL, code_responses, flatten_for_csv, load_codebook, load_reason_texts

DATA_PATH = BASE_DIR / "data" / "followup_disclaimer_phase_1" / "proposition_responses.csv"
CODEBOOK_PATH = BASE_DIR / "analysis" / "utils_files" / "preference_reason_codebook.json"
RESULTS_DIR = get_results_dir(BASE_DIR, "followup_disclaimer_phase_1", "preference_reasons")
CACHE_PATH = RESULTS_DIR / "llm_cache.jsonl"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    texts = load_reason_texts(DATA_PATH, "writer_preference_reason", "disclaimer_condition")
    codebook = load_codebook(CODEBOOK_PATH)
    coded = code_responses(texts, codebook, CACHE_PATH, model=args.model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    flatten_for_csv(coded).to_csv(RESULTS_DIR / "reason_codes.csv", index=False)
    print(f"Coded {len(coded)} of {len(texts)} responses with codebook version {codebook['version']}")


if __name__ == "__main__":
    main()
