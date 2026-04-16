#!/bin/bash
# =============================================================================
# Run All Analysis Scripts
# =============================================================================

# ==========================================
# CONFIGURATION
# ==========================================

set -euo pipefail # Exit on error, undefined variable, or pipeline failure

# Parse arguments
REGRESSION_FLAG=""

for arg in "$@"; do
    case $arg in
        --run_regressions)
            REGRESSION_FLAG="--generate_report"
            echo "Will run regressions..."
            ;;
    esac
done

# Determine repository root and paths to executables
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$REPO_ROOT/env/bin/python"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
run_regressions=true

# Check for Python executable
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python executable not found at $PYTHON_BIN"
	exit 1
fi

# Check for Rscript executable
if ! command -v "$RSCRIPT_BIN" >/dev/null 2>&1; then
	echo "Rscript executable not found: $RSCRIPT_BIN"
	exit 1
fi

# Change to repository root
cd "$REPO_ROOT"

# Helper functions to run Python scripts
run_python() {
	local script_path="$1"
	echo ">>> Running $script_path"
	"$PYTHON_BIN" "$script_path"
	echo
}

# Helper functions to run R scripts
run_r() {
	local script_path="$1"
	echo ">>> Running $script_path"
	"$RSCRIPT_BIN" "$script_path"
	echo
}

# Helper function to print section headers
print_section() {
	local title="$1"
	echo "============================================================"
	echo "$title"
	echo "============================================================"
}


# ==========================================
# EXECUTE ANALYSES
# ==========================================

# ==========================
print_section "REPOSITORY-LEVEL DATA PREP AND OVERVIEWS"
# ==========================
run_python "analysis/data_aggregation.py"
run_python "analysis/participant_overview.py"
run_r "analysis/study_overview.R"

# ==========================
print_section "MAIN STUDY: PHASE 1 (WRITING)"
# ==========================
run_python "analysis/main_phase_1/phase_1_paragraph_edits.py"
run_python "analysis/main_phase_1/phase_1_writer_engagement.py"
run_python "analysis/main_phase_1/phase_1_distortion_tolerance.py"
run_r "analysis/main_phase_1/phase_1_paragraph_preference.R"

# ==========================
print_section "MAIN STUDY: PHASE 2 (RATING - DISTRIBUTIONS)"
# ==========================
run_r "analysis/main_phase_2_distributions/phase_2_distribution_scale_variables.R"
run_r "analysis/main_phase_2_distributions/phase_2_distribution_ordinal_variables.R"
run_r "analysis/main_phase_2_distributions/phase_2_distribution_nominal_variables.R"
run_r "analysis/main_phase_2_distributions/phase_2_distribution_spread.R"
run_python "analysis/main_phase_2_distributions/phase_2_distribution_plots.py"

# ==========================
print_section "MAIN STUDY: PHASE 2 (RATING - DISTORTIONS)"
# ==========================
if [[ -n "$REGRESSION_FLAG" ]]; then
    run_r "analysis/main_phase_2_distortions/phase_2_distortion_scale_variables.R"
    run_r "analysis/main_phase_2_distortions/phase_2_distortion_ordinal_variables.R"
    run_r "analysis/main_phase_2_distortions/phase_2_distortion_nominal_variables.R"
    run_r "analysis/main_phase_2_distortions/phase_2_distortion_by_proposition_leaning.R"
fi
run_python "analysis/main_phase_2_distortions/phase_2_distortion_by_model.py"
run_python "analysis/main_phase_2_distortions/phase_2_distortion_by_input_condition.py"
run_python "analysis/main_phase_2_distortions/phase_2_distortion_plots.py"

# ==========================
print_section "DISCLAIMER STUDY: PHASE 1 (WRITING)"
# ==========================
run_python "analysis/followup_disclaimer/phase_1_distortion_tolerance.py"
run_r "analysis/followup_disclaimer/phase_1_paragraph_preference.R"

# ==========================
print_section "MITIGATION STUDY: PHASE 1 (WRITING)"
# ==========================
run_r "analysis/followup_mitigation/phase_1_paragraph_preference.R"

# ==========================
print_section "MITIGATION STUDY: PHASE 2 (RATING - DISTORTIONS)"
# ==========================
if [[ -n "$REGRESSION_FLAG" ]]; then
    run_r "analysis/followup_mitigation/phase_2_distortion_scale_variables.R"
    run_r "analysis/followup_mitigation/phase_2_distortion_ordinal_variables.R"
    run_r "analysis/followup_mitigation/phase_2_distortion_nominal_variables.R"
    run_r "analysis/followup_mitigation/phase_2_distortion_side_effects.R"
fi
run_python "analysis/followup_mitigation/phase_2_distortion_plots.py"
run_python "analysis/followup_mitigation/phase_2_mitigation_side_effects_plots.py"

# ==========================
print_section "FOLLOWUP MITIGATION: PHASE 2 (RATING - DISTRIBUTIONS)"
# ==========================
run_r "analysis/followup_mitigation/phase_2_distribution_scale_variables.R"
run_python "analysis/followup_mitigation/phase_2_distribution_plots.py"

print_section "ALL ANALYSES COMPLETED"