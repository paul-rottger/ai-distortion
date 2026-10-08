#!/usr/bin/env python3

# =============================================================================
# MITIGATION STUDY - PHASE 1 ANALYSIS: PREFERENCE REASONS (SUMMARY)
#
# Summarises LLM-coded free-text preference reasons (see
# `phase_1_preference_reasons_coding.py`). Runs offline from the committed
# `results/followup_mitigation_phase_1/preference_reasons/reason_codes.csv`.
#
# - Share of responses mentioning each reason by preferred paragraph.
# - Same by preferred paragraph and mitigation condition.
# - Distribution of number of reasons per response.
# - Dot plot in `figures/followup_mitigation_phase_1/preference_reasons.pdf`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
from pathlib import Path
import sys

# Path configuration
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "analysis" / "utils_py"))

from demo_paths import get_figures_dir, get_results_dir, parse_demo_mode
from preference_reasons_summary import summarize_preference_reasons

DEMO_MODE = parse_demo_mode()
STUDY = "followup_mitigation_phase_1"
CODEBOOK_PATH = BASE_DIR / "analysis" / "utils_files" / "preference_reason_codebook.json"
# LLM coding is not re-run in demo mode, so the committed codes are always the input
INPUT_PATH = get_results_dir(BASE_DIR, STUDY, "preference_reasons") / "reason_codes.csv"
RESULTS_DIR = get_results_dir(BASE_DIR, STUDY, "preference_reasons", demo_mode=DEMO_MODE)
FIGURE_PATH = get_figures_dir(BASE_DIR, STUDY, demo_mode=DEMO_MODE) / "preference_reasons.pdf"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    summarize_preference_reasons(INPUT_PATH, CODEBOOK_PATH, RESULTS_DIR, FIGURE_PATH)
