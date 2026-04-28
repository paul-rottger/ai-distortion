#!/usr/bin/env python3

# =============================================================================
# MAIN STUDY - PHASE 2 VISUALIZATION: DISTORTION EFFECTS
#
# Generates figure outputs for Phase 2 distortion analyses.
#
# - Loads distortion and distribution result tables across split/subset variants.
# - Produces model-, input-, and proposition-leaning comparison visualizations.
# - Saves publication-style plots to `figures/main_phase_2_distortion/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import os
import sys
from pathlib import Path

import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from adjustText import adjust_text
from scipy.stats import linregress, pearsonr, spearmanr

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "utils_py"))
REPO_ROOT = Path(BASE_DIR).resolve().parents[1]

# Internal imports
sys.path.insert(0, UTILS_DIR)
from demo_paths import get_figures_dir, get_results_dir, get_results_input_dir, parse_demo_mode
from variable_definitions import SCALE_ATTRIBUTES, ORDINAL_VARS as ORDINAL_ATTRIBUTES, NOMINAL_VARS as NOMINAL_ATTRIBUTES, MODEL_TERMS, INPUT_CONDITION_TERMS
from plotting_utils import get_group_offsets

DEMO_MODE = parse_demo_mode()
RESULTS_DIR = str(get_results_input_dir(REPO_ROOT, "main_phase_2_distortion", demo_mode=DEMO_MODE))
DISTRIBUTION_RESULTS_DIR = str(get_results_input_dir(REPO_ROOT, "main_phase_2_distribution", demo_mode=DEMO_MODE))
FIGURES_DIR = str(get_figures_dir(REPO_ROOT, "main_phase_2_distortion", demo_mode=DEMO_MODE))
DISTORTION_TOLERANCE_PATH = str(
    get_results_dir(REPO_ROOT, "main_phase_1", "distortion_responses_summary.csv", demo_mode=True)
    if DEMO_MODE
    else REPO_ROOT / "data" / "main_phase_1" / "distortion_responses_summary.csv"
)

# Plot configuration
ALL_SPLITS = ["unedited", "edited", "preferred"]
DEFAULT_PLOT_SPLITS = ["preferred"]
SUBSET_FILE_ALIASES = {
    "by_type": ["by_type"],
    "by_model": ["by_model"],
    "by_input": ["by_input", "by_input_condition"],
}
PROPOSITION_LEANINGS = ["left", "right"]
PROPOSITION_LEANING_SPLIT = "preferred"
PROPOSITION_LEANING_SUBSET = "by_proposition_leaning"

MODEL_TERM_ORDER = MODEL_TERMS

INPUT_TERM_ORDER = INPUT_CONDITION_TERMS

TERM_LABELS = {
    "model_anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "model_deepseek/deepseek-chat-v3-0324": "DeepSeek Chat V3",
    "model_openai/chatgpt-4o-latest": "ChatGPT 4o",
    "input_condition_stance-based": "Stance-based",
    "input_condition_bullets-based": "Bullets-based",
    "input_condition_rewrite": "Rewrite",
    "input_condition_improve": "Improve",
    "left": "Left-leaning",
    "right": "Right-leaning",
}

TERM_COLORS = {
    "model_anthropic/claude-sonnet-4": "#1B4965",
    "model_deepseek/deepseek-chat-v3-0324": "#CA6702",
    "model_openai/chatgpt-4o-latest": "#2A9D8F",
    "input_condition_stance-based": "#0101FF",
    "input_condition_bullets-based": "#44239F",
    "input_condition_rewrite": "#B90ED3",
    "input_condition_improve": "#E844BF",
    "left": "#C84C09",
    "right": "#1F6AA5",
}

TERM_ORDERS = {
    "by_model": MODEL_TERM_ORDER,
    "by_input": INPUT_TERM_ORDER,
    PROPOSITION_LEANING_SUBSET: PROPOSITION_LEANINGS,
}

SCALE_GROUP_BOUNDARIES = [2.5, 4.5, 9.5, 14.5]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_results_by_attribute(attributes, subset_names, splits=None, results_dir=RESULTS_DIR):
    if splits is None:
        splits = ALL_SPLITS

    loaded_results = {}

    for para in splits:
        split_results = {}
        for attr in reversed(attributes):
            attr_results = {}
            for subset in subset_names:
                candidate_subsets = SUBSET_FILE_ALIASES.get(subset, [subset])
                for candidate_subset in candidate_subsets:
                    path = os.path.join(results_dir, para, f"{attr}_{candidate_subset}.csv")
                    if os.path.exists(path):
                        attr_results[subset] = pd.read_csv(path)
                        break

            if attr_results:
                split_results[attr] = attr_results

        if split_results:
            loaded_results[para] = split_results

    return loaded_results


def load_results_by_proposition_leaning(attributes, results_dir=RESULTS_DIR):
    loaded_results = {PROPOSITION_LEANING_SPLIT: {}}
    base_dir = os.path.join(
        results_dir,
        PROPOSITION_LEANING_SPLIT,
        PROPOSITION_LEANING_SUBSET,
    )

    for attr in reversed(attributes):
        leaning_frames = []
        for leaning in PROPOSITION_LEANINGS:
            path = os.path.join(base_dir, f"{attr}_{leaning}.csv")
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if df.empty:
                continue

            df = df.copy()
            df["term"] = leaning
            leaning_frames.append(df)

        if leaning_frames:
            loaded_results[PROPOSITION_LEANING_SPLIT][attr] = {
                PROPOSITION_LEANING_SUBSET: pd.concat(leaning_frames, ignore_index=True)
            }

    if not loaded_results[PROPOSITION_LEANING_SPLIT]:
        return {}

    return loaded_results


# =============================================================================
# OUTPUTS
# =============================================================================

def build_split_figure_path(split, filename):
    return os.path.join(FIGURES_DIR, split, filename)


def save_figure(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


# =============================================================================
# ANALYSIS
# =============================================================================

def split_has_required_columns(
    results_dict,
    split,
    attributes,
    subset,
    required_columns,
):
    if split not in results_dict:
        return False

    for attr in attributes:
        attr_results = results_dict[split].get(attr, {})
        if subset not in attr_results:
            return False

        df = attr_results[subset]
        if df.empty or any(column not in df.columns for column in required_columns):
            return False

    return True


def prettify_term_label(term):
    return TERM_LABELS.get(term, term)


def get_available_attributes(
    results_dict,
    split,
    attributes,
    subset,
    required_columns,
):
    if split not in results_dict:
        return []

    available_attributes = []
    for attr in attributes:
        attr_results = results_dict[split].get(attr, {})
        if subset not in attr_results:
            continue

        df = attr_results[subset]
        if df.empty or any(column not in df.columns for column in required_columns):
            continue

        available_attributes.append(attr)

    return available_attributes


def get_term_order(subset, terms):
    expected_terms = TERM_ORDERS.get(subset, [])
    ordered_terms = [term for term in expected_terms if term in terms]
    extra_terms = sorted(term for term in terms if term not in expected_terms)
    return ordered_terms + extra_terms


def get_available_terms(
    results_dict,
    split,
    attributes,
    subset,
    required_columns,
    candidate_attributes=None,
):
    term_sets = []
    target_attributes = candidate_attributes or attributes

    if split not in results_dict:
        return []

    for attr in target_attributes:
        attr_results = results_dict[split].get(attr, {})
        if subset not in attr_results:
            return []

        df = attr_results[subset]
        if df.empty or any(column not in df.columns for column in required_columns):
            return []

        terms = df["term"].dropna().unique().tolist()
        if not terms:
            return []
        term_sets.append(set(terms))

    common_terms = set.intersection(*term_sets)
    return get_term_order(subset, common_terms)


def create_horizontal_grouped_effect_plot(
    regression_dict,
    attributes,
    split,
    subset,
    estimate_column,
    lower_column,
    upper_column,
    figsize,
    xlabel,
    ylabel,
    reference_line,
    save_path=None,
    group_boundaries=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    available_attributes = get_available_attributes(
        regression_dict,
        split,
        attributes,
        subset,
        ["term", estimate_column, lower_column, upper_column],
    )
    if not available_attributes:
        raise ValueError(f"No plottable attributes are available for split '{split}' and subset '{subset}'.")

    included_terms = get_available_terms(
        regression_dict,
        split,
        attributes,
        subset,
        ["term", estimate_column, lower_column, upper_column],
        candidate_attributes=available_attributes,
    )
    if not included_terms:
        raise ValueError(f"No requested terms are available for split '{split}' and subset '{subset}'.")

    colors = {term: TERM_COLORS[term] for term in included_terms}
    y_offset = get_group_offsets(included_terms)
    attribute_order = list(reversed(available_attributes))
    attribute_positions = {
        attribute: index for index, attribute in enumerate(attribute_order)
    }

    for term in included_terms:
        term_frames = []
        for attr in attribute_order:
            df = regression_dict[split].get(attr, {}).get(subset)
            if df is None:
                continue

            term_df = df[df["term"] == term].copy()
            if term_df.empty:
                continue

            term_df["outcome"] = attr
            term_frames.append(term_df)

        if not term_frames:
            continue

        regression_df = pd.concat(term_frames, ignore_index=True)
        y_positions = [
            attribute_positions[outcome] + y_offset[term]
            for outcome in regression_df["outcome"]
        ]

        ax.errorbar(
            regression_df[estimate_column],
            y_positions,
            xerr=[
                regression_df[estimate_column] - regression_df[lower_column],
                regression_df[upper_column] - regression_df[estimate_column],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[term],
            label=prettify_term_label(term),
            zorder=4,
        )

    ax.axvline(reference_line, color="black", linestyle=(0, (5, 7)), linewidth=1)

    if group_boundaries:
        for boundary in group_boundaries:
            ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    y_tick_positions = list(range(len(attribute_order)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(attribute_order)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if len(included_terms) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


def prepare_nominal_regression_df_for_term(
    regression_dict,
    para_type,
    term,
    subset="by_type",
    attributes=None,
):
    rows = []
    nominal_attributes = attributes or NOMINAL_ATTRIBUTES

    for attr in reversed(nominal_attributes):
        if attr not in regression_dict.get(para_type, {}):
            continue
        if subset not in regression_dict[para_type][attr]:
            continue

        df = regression_dict[para_type][attr][subset].copy()
        if df.empty:
            continue

        df = df[df["term"] == term].copy()
        if df.empty:
            continue

        df["outcome"] = attr
        df["comparison_label"] = df.apply(
            lambda row: f"{row['outcome']}: {row['target_level']} vs {row['reference_level']}",
            axis=1,
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def get_nominal_comparison_order_for_terms(
    regression_dict,
    para_type,
    included_terms,
    subset="by_type",
    attributes=None,
):
    comparison_order = []
    seen = set()
    nominal_attributes = attributes or NOMINAL_ATTRIBUTES

    for attr in reversed(nominal_attributes):
        for term in included_terms:
            if attr not in regression_dict.get(para_type, {}):
                continue
            if subset not in regression_dict[para_type][attr]:
                continue

            df = regression_dict[para_type][attr][subset]
            df = df[df["term"] == term]
            for _, row in df.iterrows():
                label = f"{attr}: {row['target_level']} vs {row['reference_level']}"
                if label not in seen:
                    seen.add(label)
                    comparison_order.append(label)

    return comparison_order


def create_horizontal_odds_ratio_plot_nominal_grouped(
    regression_dict,
    split,
    subset="by_model",
    figsize=(10, 10),
    xlabel="Odds Ratio for AI vs. Writer Paragraphs by Category.\n1 = No Difference.",
    ylabel="Nominal Attribute Categories",
    save_path=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    available_attributes = get_available_attributes(
        regression_dict,
        split,
        NOMINAL_ATTRIBUTES,
        subset,
        ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
    )
    if not available_attributes:
        raise ValueError(f"No plottable attributes are available for split '{split}' and subset '{subset}'.")

    included_terms = get_available_terms(
        regression_dict,
        split,
        NOMINAL_ATTRIBUTES,
        subset,
        ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
        candidate_attributes=available_attributes,
    )
    if not included_terms:
        raise ValueError(f"No requested terms are available for split '{split}' and subset '{subset}'.")

    comparison_order = get_nominal_comparison_order_for_terms(
        regression_dict,
        split,
        included_terms,
        subset=subset,
        attributes=available_attributes,
    )
    group_boundaries = get_nominal_group_boundaries(comparison_order)
    comparison_positions = {
        label: index for index, label in enumerate(comparison_order)
    }
    y_offset = get_group_offsets(included_terms)

    for term in included_terms:
        regression_df = prepare_nominal_regression_df_for_term(
            regression_dict,
            split,
            term,
            subset=subset,
            attributes=available_attributes,
        )

        if regression_df.empty:
            continue

        y_positions = [
            comparison_positions[label] + y_offset[term]
            for label in regression_df["comparison_label"]
        ]

        ax.errorbar(
            regression_df["odds_ratio"],
            y_positions,
            xerr=[
                regression_df["odds_ratio"] - regression_df["or_low"],
                regression_df["or_high"] - regression_df["odds_ratio"],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=TERM_COLORS[term],
            label=prettify_term_label(term),
            zorder=4,
        )

    ax.axvline(1, color="black", linestyle=(0, (5, 7)), linewidth=1)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    y_tick_positions = list(range(len(comparison_order)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(comparison_order)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if len(included_terms) > 1:
        ax.legend(frameon=False, loc="lower left")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


# =============================================================================
# MAIN EXECUTION
# =============================================================================

################################
# SCALE ATTRIBUTES - AME PLOTS
################################

regression_dict = load_results_by_attribute(
    SCALE_ATTRIBUTES,
    subset_names=["by_type", "by_model", "by_input"],
)

proposition_regression_dict = load_results_by_proposition_leaning(SCALE_ATTRIBUTES)


# Create horizontal AME plot for "by_type" subset
def create_horizontal_ame_plot(
    regression_dict,
    subset="by_type",
    included_splits=None,
    figsize=(8, 10),
    xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nLarger = More Distortion from AI.",
    ylabel="Scale Attributes (Grouped by Type)",
    save_path=None,
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    if len(included_splits) == 1:
        y_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        y_offset = {
            para_type: position * 0.12
            for para_type, position in zip(included_splits, offset_positions)
        }

    for para_type in included_splits:
        if para_type in regression_dict:
            # Select the appropriate dataframe based on subset
            regression_df = pd.concat(
                [
                    df.assign(outcome=attr)
                    for attr, df in (
                        (attr, regression_dict[para_type][attr][subset])
                        for attr in regression_dict[para_type]
                    )
                ],
                ignore_index=True,
            )

            # Create y-positions with offset
            y_positions = [
                list(regression_dict[para_type].keys()).index(outcome)
                + y_offset[para_type]
                for outcome in regression_df["outcome"]
            ]

            # Plot horizontal error bars
            ax.errorbar(
                regression_df["ame"],
                y_positions,
                xerr=[
                    regression_df["ame"] - regression_df["ame_low"],
                    regression_df["ame_high"] - regression_df["ame"],
                ],
                fmt="o",
                capsize=3,
                capthick=1,
                elinewidth=1,
                color=colors[para_type],
                label=para_type.capitalize(),
                zorder=4,
            )

    # Add reference line at x=0
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Add vertical lines to separate attribute groups
    group_boundaries = [2.5, 4.5, 9.5, 14.5]
    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(SCALE_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(SCALE_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="lightgray", linestyle="-", linewidth=0.5)

    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_figure(fig, save_path)

    return fig, ax


available_splits = [
    split
    for split in ALL_SPLITS
    if split_has_required_columns(
        regression_dict,
        split,
        SCALE_ATTRIBUTES,
        "by_type",
        ["ame", "ame_low", "ame_high"],
    )
]

missing_splits = [split for split in ALL_SPLITS if split not in available_splits]
for split in missing_splits:
    print(f"Skipping scale AME data for split '{split}' due to missing results.")

if available_splits:
    fig, _ = create_horizontal_ame_plot(
        regression_dict,
        included_splits=available_splits,
        ylabel=None,
        save_path=os.path.join(FIGURES_DIR, "distortion_scale_variables_ame.pdf"),
    )
    plt.close(fig)
else:
    print("Skipping scale AME plot because no compatible split results were found.")

for split in ALL_SPLITS:
    for subset, file_suffix in [("by_model", "by_model"), ("by_input", "by_input")]:
        available_attributes = get_available_attributes(
            regression_dict,
            split,
            SCALE_ATTRIBUTES,
            subset,
            ["term", "ame", "ame_low", "ame_high"],
        )
        if not available_attributes:
            print(
                f"Skipping scale AME plot for split '{split}' and subset '{subset}' due to missing results."
            )
            continue

        fig, _ = create_horizontal_grouped_effect_plot(
            regression_dict,
            attributes=SCALE_ATTRIBUTES,
            split=split,
            subset=subset,
            estimate_column="ame",
            lower_column="ame_low",
            upper_column="ame_high",
            figsize=(8, 10),
            xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nLarger = More Distortion from AI.",
            ylabel=None,
            reference_line=0,
            save_path=build_split_figure_path(
                split,
                f"distortion_scale_variables_ame_{file_suffix}.pdf",
            ),
            group_boundaries=SCALE_GROUP_BOUNDARIES,
        )
        plt.close(fig)

proposition_available_attributes = get_available_attributes(
    proposition_regression_dict,
    PROPOSITION_LEANING_SPLIT,
    SCALE_ATTRIBUTES,
    PROPOSITION_LEANING_SUBSET,
    ["term", "ame", "ame_low", "ame_high"],
)

if proposition_available_attributes:
    fig, _ = create_horizontal_grouped_effect_plot(
        proposition_regression_dict,
        attributes=SCALE_ATTRIBUTES,
        split=PROPOSITION_LEANING_SPLIT,
        subset=PROPOSITION_LEANING_SUBSET,
        estimate_column="ame",
        lower_column="ame_low",
        upper_column="ame_high",
        figsize=(8, 10),
        xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nLarger = More Distortion from AI.",
        ylabel=None,
        reference_line=0,
        save_path=build_split_figure_path(
            PROPOSITION_LEANING_SPLIT,
            "distortion_scale_variables_ame_by_proposition_leaning.pdf",
        ),
        group_boundaries=SCALE_GROUP_BOUNDARIES,
    )
    plt.close(fig)
else:
    print("Skipping scale AME by proposition leaning plot due to missing results.")

################################
# SCALE ATTRIBUTES - DISTORTION VS AVG WRITER TOLERANCE
################################

OUTCOME_MAP_SCALE = {
    "writer_stance_polarity": ("extreme_more", "moderate_more"),
    "writer_confidence": ("confidence_more", "confidence_less"),
    "writer_knowledge": ("knowledge_more", "knowledge_less"),
    "writer_importance": ("importance_more", "importance_less"),
    "paragraph_relevance": ("relevance_more", "relevance_less"),
    "paragraph_clarity": ("clarity_more", "clarity_less"),
    "paragraph_formality": ("formality_more", "formality_less"),
    "paragraph_informativeness": ("informative_more", "informative_less"),
    "paragraph_originality": ("originality_more", "originality_less"),
    "writer_friendliness": ("friendliness_more", "friendliness_less"),
    "writer_optimism": ("optimism_more", "optimism_less"),
    "writer_community": ("community_more", "community_less"),
    "writer_openness": ("open_views_more", "open_views_less"),
    "paragraph_hope": ("hope_more", "hope_less"),
    "paragraph_excitement": ("excitement_more", "excitement_less"),
    "paragraph_fear": ("fear_more", "fear_less"),
    "paragraph_disgust": ("disgust_more", "disgust_less"),
    "paragraph_anger": ("anger_more", "anger_less"),
}


def match_outcome_distortion_scale(row):
    labels = OUTCOME_MAP_SCALE.get(row.outcome)
    if labels is None or row.ame == 0:
        return None
    return labels[0] if row.ame > 0 else labels[1]


distortion_tolerance_df = pd.read_csv(DISTORTION_TOLERANCE_PATH)


def create_ame_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    included_splits=None,
    filter_term=None,
    figsize=(9, 6),
    s=30,
    title=None,
    xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Average Tolerance Score (0–100).\nHigher = More Accepting of This Distortion.",
    save_path=None,
    annotate_points=True,
):

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------
    # Colors + Offsets
    # --------------------------------
    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    if len(included_splits) == 1:
        x_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        x_offset = {
            para_type: position * 0.005
            for para_type, position in zip(included_splits, offset_positions)
        }

    # --------------------------------
    # Process conditions
    # --------------------------------
    for para_type in included_splits:
        if para_type not in regression_dict:
            continue

        regression_df = pd.concat(
            [
                df.assign(outcome=attr)
                for attr, df in (
                    (attr, regression_dict[para_type][attr][subset])
                    for attr in regression_dict[para_type]
                )
            ],
            ignore_index=True,
        )

        if filter_term:
            regression_df = regression_df[
                regression_df["term"].str.contains(filter_term)
            ]

        regression_df["distortion"] = regression_df.apply(
            match_outcome_distortion_scale, axis=1
        )

        merged_df = regression_df.merge(
            distortion_tolerance_df,
            on="distortion",
            how="inner",
        )

        if merged_df.empty:
            continue

        # Apply small horizontal offset
        x_values = merged_df["ame"] + x_offset[para_type]
        y_values = merged_df["mean"]

        # Scatter
        ax.scatter(
            x_values,
            y_values,
            s=s,
            color=colors[para_type],
            edgecolors="black",
            linewidth=1,
            label=para_type.capitalize(),
            zorder=5,
        )

        # Error bars
        x_err = [
            merged_df["ame"] - merged_df["ame_low"],
            merged_df["ame_high"] - merged_df["ame"],
        ]

        y_err = [
            merged_df["mean"] - merged_df["mean_ci_low"],
            merged_df["mean_ci_high"] - merged_df["mean"],
        ]

        ax.errorbar(
            x_values,
            y_values,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[para_type],
            zorder=4,
        )

        if len(merged_df) >= 2 and x_values.nunique() > 1:
            fit = linregress(x_values, y_values)
            x_fit = pd.Series([x_values.min(), x_values.max()])
            y_fit = fit.intercept + fit.slope * x_fit
            ax.plot(
                x_fit,
                y_fit,
                color=colors[para_type],
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )

        # Annotations
        if annotate_points:
            texts = []
            for _, row in merged_df.iterrows():
                text = ax.annotate(
                    row["outcome"],
                    (row["ame"] + x_offset[para_type], row["mean"]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="square,pad=0.25",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.85,
                    ),
                    zorder=6,
                )
                texts.append(text)

            adjust_text(
                texts,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.7,
                    lw=0.5,
                ),
                force_static=(10, 10),
                force_text=(1, 1),
            )

    # --------------------------------
    # Background tolerance bands
    # --------------------------------
    x_min, x_max = ax.get_xlim()

    band_colors = {
        "0_25": "#cc0000",
        "25_50": "#e06666",
        "50_75": "#93c47d",
        "75_100": "#38761d",
    }

    ax.add_patch(
        Rectangle(
            (x_min, 0),
            x_max - x_min,
            25,
            facecolor=band_colors["0_25"],
            alpha=0.25,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_min, 25),
            x_max - x_min,
            25,
            facecolor=band_colors["25_50"],
            alpha=0.25,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_min, 50),
            x_max - x_min,
            25,
            facecolor=band_colors["50_75"],
            alpha=0.25,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_min, 75),
            x_max - x_min,
            25,
            facecolor=band_colors["75_100"],
            alpha=0.25,
            zorder=0,
        )
    )

    # --------------------------------
    # Reference lines + formatting
    # --------------------------------
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


for split in ALL_SPLITS:
    if not split_has_required_columns(
        regression_dict,
        split,
        SCALE_ATTRIBUTES,
        "by_type",
        ["ame", "ame_low", "ame_high"],
    ):
        print(
            f"Skipping scale tolerance scatter for split '{split}' due to missing results."
        )
        continue

    fig, _ = create_ame_tolerance_scatterplot(
        regression_dict,
        distortion_tolerance_df,
        subset="by_type",
        included_splits=[split],
        save_path=build_split_figure_path(split, "distortion_scale_variables_tolerance_scatter.pdf"),
    )
    plt.close(fig)


def calculate_ame_tolerance_correlations(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    included_splits=None,
    filter_term=None,
):
    """
    Calculates Pearson and Spearman correlations between:
        X = AME (AI vs Writer distortion)
        Y = Mean tolerance score

    Returns dictionary of results and the merged data used.
    """

    results = {}
    combined_df = []

    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    para_types = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    for para_type in para_types:
        if para_type not in regression_dict:
            continue

        # ----------------------------
        # Reconstruct plotted dataset
        # ----------------------------
        regression_df = pd.concat(
            [
                df.assign(outcome=attr)
                for attr, df in (
                    (attr, regression_dict[para_type][attr][subset])
                    for attr in regression_dict[para_type]
                )
            ],
            ignore_index=True,
        )

        if filter_term:
            regression_df = regression_df[
                regression_df["term"].str.contains(filter_term)
            ]

        regression_df["distortion"] = regression_df.apply(
            match_outcome_distortion_scale, axis=1
        )

        merged_df = regression_df.merge(
            distortion_tolerance_df,
            on="distortion",
            how="inner",
        )

        if merged_df.empty:
            continue

        # Drop missing
        merged_df = merged_df.dropna(subset=["ame", "mean"])

        x = merged_df["ame"]
        y = merged_df["mean"]

        # ----------------------------
        # Correlations
        # ----------------------------
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)

        results[para_type] = {
            "n": len(merged_df),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        }

        merged_df["condition"] = para_type
        combined_df.append(merged_df)

    return results

correlation_results = calculate_ame_tolerance_correlations(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
)

print("Correlation Results:", correlation_results)


################################
# ORDINAL ATTRIBUTES
################################

# Load regression results for each attribute and split
regression_dict = load_results_by_attribute(
    ORDINAL_ATTRIBUTES,
    subset_names=["by_type", "by_model", "by_input"],
)

proposition_regression_dict = load_results_by_proposition_leaning(ORDINAL_ATTRIBUTES)


# Create horizontal odds ratio plot for "by_type" subset
def create_horizontal_odds_ratio_plot(
    regression_dict,
    subset="by_type",
    included_splits=None,
    figsize=(8, 4),
    xlabel="Odds Ratio for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Ordinal Attributes",
    save_path=None,
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each condition
    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    # Define y-position offsets to avoid overlapping points
    if len(included_splits) == 1:
        y_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        y_offset = {
            para_type: position * 0.12
            for para_type, position in zip(included_splits, offset_positions)
        }

    # Process both unedited and edited data
    for para_type in included_splits:
        if para_type in regression_dict:
            # Select the appropriate dataframe based on subset
            regression_df = pd.concat(
                [
                    df.assign(outcome=attr)
                    for attr, df in (
                        (attr, regression_dict[para_type][attr][subset])
                        for attr in regression_dict[para_type]
                    )
                ],
                ignore_index=True,
            )

            # Create y-positions with offset
            y_positions = [
                list(regression_dict[para_type].keys()).index(outcome)
                + y_offset[para_type]
                for outcome in regression_df["outcome"]
            ]

            # Plot horizontal error bars
            ax.errorbar(
                regression_df["odds_ratio"],
                y_positions,
                xerr=[
                    regression_df["odds_ratio"] - regression_df["or_low"],
                    regression_df["or_high"] - regression_df["odds_ratio"],
                ],
                fmt="o",
                capsize=3,
                capthick=1,
                elinewidth=1,
                color=colors[para_type],
                label=para_type.capitalize(),
                zorder=4,
            )

    # Add reference line at x=0
    ax.axvline(1, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(ORDINAL_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(ORDINAL_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_figure(fig, save_path)

    return fig, ax


available_splits = [
    split
    for split in ALL_SPLITS
    if split_has_required_columns(
        regression_dict,
        split,
        ORDINAL_ATTRIBUTES,
        "by_type",
        ["odds_ratio", "or_low", "or_high"],
    )
]

missing_splits = [split for split in ALL_SPLITS if split not in available_splits]
for split in missing_splits:
    print(
        f"Skipping ordinal odds-ratio data for split '{split}' due to incompatible results."
    )

if available_splits:
    fig, _ = create_horizontal_odds_ratio_plot(
        regression_dict,
        included_splits=available_splits,
        ylabel=None,
        save_path=os.path.join(FIGURES_DIR, "distortion_ordinal_variables_odds_ratio.pdf"),
    )
    plt.close(fig)
else:
    print("Skipping ordinal odds-ratio plot because no compatible split results were found.")

for split in ALL_SPLITS:
    for subset, file_suffix in [("by_model", "by_model"), ("by_input", "by_input")]:
        available_attributes = get_available_attributes(
            regression_dict,
            split,
            ORDINAL_ATTRIBUTES,
            subset,
            ["term", "odds_ratio", "or_low", "or_high"],
        )
        if not available_attributes:
            print(
                f"Skipping ordinal odds-ratio plot for split '{split}' and subset '{subset}' due to missing results."
            )
            continue

        fig, _ = create_horizontal_grouped_effect_plot(
            regression_dict,
            attributes=ORDINAL_ATTRIBUTES,
            split=split,
            subset=subset,
            estimate_column="odds_ratio",
            lower_column="or_low",
            upper_column="or_high",
            figsize=(8, 4),
            xlabel="Odds Ratio for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
            ylabel=None,
            reference_line=1,
            save_path=build_split_figure_path(
                split,
                f"distortion_ordinal_variables_odds_ratio_{file_suffix}.pdf",
            ),
        )
        plt.close(fig)

proposition_available_attributes = get_available_attributes(
    proposition_regression_dict,
    PROPOSITION_LEANING_SPLIT,
    ORDINAL_ATTRIBUTES,
    PROPOSITION_LEANING_SUBSET,
    ["term", "odds_ratio", "or_low", "or_high"],
)

if proposition_available_attributes:
    fig, _ = create_horizontal_grouped_effect_plot(
        proposition_regression_dict,
        attributes=ORDINAL_ATTRIBUTES,
        split=PROPOSITION_LEANING_SPLIT,
        subset=PROPOSITION_LEANING_SUBSET,
        estimate_column="odds_ratio",
        lower_column="or_low",
        upper_column="or_high",
        figsize=(8, 4),
        xlabel="Odds Ratio for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
        ylabel=None,
        reference_line=1,
        save_path=build_split_figure_path(
            PROPOSITION_LEANING_SPLIT,
            "distortion_ordinal_variables_odds_ratio_by_proposition_leaning.pdf",
        ),
    )
    plt.close(fig)
else:
    print("Skipping ordinal odds-ratio by proposition leaning plot due to missing results.")


################################
# ORDINAL ATTRIBUTES - DISTORTION VS WRITER TOLERANCE
################################

OUTCOME_MAP_ORDINAL = {
    "writer_education": (
        "education_more",
        "education_less",
    ),
    "writer_english_skills": (
        "english_better",
        "english_worse",
    ),
    "writer_income": (
        "income_higher",
        "income_lower",
    ),
    "writer_age_binned": (
        "age_older",
        "age_younger",
    ),
}


def match_outcome_distortion_ordinal(row):
    labels = OUTCOME_MAP_ORDINAL.get(row.outcome)
    if labels is None or row.odds_ratio == 1:
        return None
    return labels[0] if row.odds_ratio > 1 else labels[1]


def create_oddsratio_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    included_splits=None,
    filter_term=None,
    figsize=(9, 6),
    s=40,
    title=None,
    xlabel="Odds Ratio for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Average Tolerance Score (0–100).\nHigher = More Accepting of This Distortion.",
    save_path=None,
    annotate_points=True,
):

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------
    # Colors + Offsets
    # --------------------------------
    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    if len(included_splits) == 1:
        x_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        x_offset = {
            para_type: position * 0.01
            for para_type, position in zip(included_splits, offset_positions)
        }

    # --------------------------------
    # Process conditions
    # --------------------------------
    for para_type in included_splits:
        if para_type not in regression_dict:
            continue

        regression_df = pd.concat(
            [
                df.assign(outcome=attr)
                for attr, df in (
                    (attr, regression_dict[para_type][attr][subset])
                    for attr in regression_dict[para_type]
                )
            ],
            ignore_index=True,
        )

        if filter_term:
            regression_df = regression_df[
                regression_df["term"].str.contains(filter_term)
            ]

        regression_df["distortion"] = regression_df.apply(
            match_outcome_distortion_ordinal, axis=1
        )

        merged_df = regression_df.merge(
            distortion_tolerance_df,
            on="distortion",
            how="inner",
        )

        if merged_df.empty:
            continue

        # X/Y values
        x_values = merged_df["odds_ratio"] + x_offset[para_type]
        y_values = merged_df["mean"]

        # Scatter
        ax.scatter(
            x_values,
            y_values,
            s=s,
            color=colors[para_type],
            edgecolors="black",
            linewidth=1,
            label=para_type.capitalize(),
            zorder=5,
        )

        # Error bars
        x_err = [
            merged_df["odds_ratio"] - merged_df["or_low"],
            merged_df["or_high"] - merged_df["odds_ratio"],
        ]

        y_err = [
            merged_df["mean"] - merged_df["mean_ci_low"],
            merged_df["mean_ci_high"] - merged_df["mean"],
        ]

        ax.errorbar(
            x_values,
            y_values,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[para_type],
            zorder=4,
        )

        # --------------------------------
        # Annotations
        # --------------------------------
        if annotate_points:
            texts = []
            for _, row in merged_df.iterrows():
                text = ax.annotate(
                    row["outcome"],
                    (row["odds_ratio"] + x_offset[para_type], row["mean"]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="square,pad=0.25",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.85,
                    ),
                    zorder=6,
                )
                texts.append(text)

            adjust_text(
                texts,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.7,
                    lw=0.5,
                ),
                force_static=(10, 10),
                force_text=(1, 1),
            )

    ax.axvline(1, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # --------------------------------
    # Background tolerance bands
    # --------------------------------
    x_min, x_max = ax.get_xlim()

    band_colors = {
        "0_25": "#cc0000",
        "25_50": "#e06666",
        "50_75": "#93c47d",
        "75_100": "#38761d",
    }

    for lower, color_key in zip(
        [0, 25, 50, 75],
        ["0_25", "25_50", "50_75", "75_100"],
    ):
        ax.add_patch(
            Rectangle(
                (x_min, lower),
                x_max - x_min,
                25,
                facecolor=band_colors[color_key],
                alpha=0.25,
                zorder=0,
            )
        )

    # --------------------------------
    # Formatting
    # --------------------------------

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


for split in ALL_SPLITS:
    if not split_has_required_columns(
        regression_dict,
        split,
        ORDINAL_ATTRIBUTES,
        "by_type",
        ["odds_ratio", "or_low", "or_high"],
    ):
        print(
            f"Skipping ordinal tolerance scatter for split '{split}' due to incompatible results."
        )
        continue

    fig, _ = create_oddsratio_tolerance_scatterplot(
        regression_dict,
        distortion_tolerance_df,
        subset="by_type",
        included_splits=[split],
        save_path=build_split_figure_path(split, "distortion_ordinal_variables_tolerance_scatter.pdf"),
    )
    plt.close(fig)


################################
# NOMINAL ATTRIBUTES
################################

# Load regression results for each attribute and split
regression_dict = load_results_by_attribute(
    NOMINAL_ATTRIBUTES,
    subset_names=["by_type", "by_model", "by_input"],
)

proposition_regression_dict = load_results_by_proposition_leaning(NOMINAL_ATTRIBUTES)

nominal_distribution_dict = load_results_by_attribute(
    NOMINAL_ATTRIBUTES,
    subset_names=["by_type"],
    results_dir=DISTRIBUTION_RESULTS_DIR,
)


def prepare_nominal_regression_df(regression_dict, para_type, subset="by_type"):
    rows = []

    for attr in reversed(NOMINAL_ATTRIBUTES):
        if attr not in regression_dict.get(para_type, {}):
            continue
        if subset not in regression_dict[para_type][attr]:
            continue

        df = regression_dict[para_type][attr][subset].copy()
        if df.empty:
            continue

        df["outcome"] = attr
        df["comparison_label"] = df.apply(
            lambda row: f"{row['outcome']}: {row['target_level']} vs {row['reference_level']}",
            axis=1,
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def prepare_nominal_cramers_df(regression_dict, para_type, subset="by_type"):
    rows = []

    for attr in reversed(NOMINAL_ATTRIBUTES):
        if attr not in regression_dict.get(para_type, {}):
            continue
        if subset not in regression_dict[para_type][attr]:
            continue

        df = regression_dict[para_type][attr][subset].copy()
        if df.empty:
            continue

        df["outcome"] = attr
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def get_nominal_comparison_order(regression_dict, included_splits, subset="by_type"):
    comparison_order = []
    seen = set()

    for attr in reversed(NOMINAL_ATTRIBUTES):
        for para_type in included_splits:
            if attr not in regression_dict.get(para_type, {}):
                continue
            if subset not in regression_dict[para_type][attr]:
                continue

            df = regression_dict[para_type][attr][subset]
            for _, row in df.iterrows():
                label = f"{attr}: {row['target_level']} vs {row['reference_level']}"
                if label not in seen:
                    seen.add(label)
                    comparison_order.append(label)

    return comparison_order


def get_nominal_group_boundaries(comparison_order):
    group_boundaries = []
    previous_attribute = None

    for index, label in enumerate(comparison_order):
        current_attribute = label.split(":", 1)[0]
        if previous_attribute is not None and current_attribute != previous_attribute:
            group_boundaries.append(index - 0.5)
        previous_attribute = current_attribute

    return group_boundaries


def create_horizontal_odds_ratio_plot_nominal(
    regression_dict,
    subset="by_type",
    included_splits=None,
    figsize=(10, 10),
    xlabel="Odds Ratio for AI vs. Writer Paragraphs by Category.\n1 = No Difference.",
    ylabel="Nominal Attribute Categories",
    save_path=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    comparison_order = get_nominal_comparison_order(
        regression_dict,
        included_splits,
        subset=subset,
    )
    group_boundaries = get_nominal_group_boundaries(comparison_order)
    comparison_positions = {
        label: index for index, label in enumerate(comparison_order)
    }

    # Offset
    if len(included_splits) == 1:
        y_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        y_offset = {
            para_type: position * 0.12
            for para_type, position in zip(included_splits, offset_positions)
        }

    for para_type in included_splits:
        if para_type not in regression_dict:
            continue

        regression_df = prepare_nominal_regression_df(
            regression_dict,
            para_type,
            subset=subset,
        )

        if regression_df.empty:
            continue

        y_positions = [
            comparison_positions[label] + y_offset[para_type]
            for label in regression_df["comparison_label"]
        ]

        ax.errorbar(
            regression_df["odds_ratio"],
            y_positions,
            xerr=[
                regression_df["odds_ratio"] - regression_df["or_low"],
                regression_df["or_high"] - regression_df["odds_ratio"],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[para_type],
            label=para_type.capitalize(),
            zorder=4,
        )

    ax.axvline(1, color="black", linestyle=(0, (5, 7)), linewidth=1)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    y_tick_positions = list(range(len(comparison_order)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(comparison_order)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower left")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


available_splits = [
    split
    for split in ALL_SPLITS
    if split_has_required_columns(
        regression_dict,
        split,
        NOMINAL_ATTRIBUTES,
        "by_type",
        ["odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
    )
]

missing_splits = [split for split in ALL_SPLITS if split not in available_splits]
for split in missing_splits:
    print(
        f"Skipping nominal odds-ratio data for split '{split}' due to incompatible results."
    )

if available_splits:
    fig, _ = create_horizontal_odds_ratio_plot_nominal(
        regression_dict,
        included_splits=available_splits,
        ylabel=None,
        save_path=os.path.join(FIGURES_DIR, "distortion_nominal_variables_odds_ratio.pdf"),
    )
    plt.close(fig)
else:
    print("Skipping nominal odds-ratio plot because no compatible split results were found.")

for split in ALL_SPLITS:
    for subset, file_suffix in [("by_model", "by_model"), ("by_input", "by_input")]:
        available_attributes = get_available_attributes(
            regression_dict,
            split,
            NOMINAL_ATTRIBUTES,
            subset,
            ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
        )
        if not available_attributes:
            print(
                f"Skipping nominal odds-ratio plot for split '{split}' and subset '{subset}' due to missing results."
            )
            continue

        fig, _ = create_horizontal_odds_ratio_plot_nominal_grouped(
            regression_dict,
            split=split,
            subset=subset,
            ylabel=None,
            save_path=build_split_figure_path(
                split,
                f"distortion_nominal_variables_odds_ratio_{file_suffix}.pdf",
            ),
        )
        plt.close(fig)

proposition_available_attributes = get_available_attributes(
    proposition_regression_dict,
    PROPOSITION_LEANING_SPLIT,
    NOMINAL_ATTRIBUTES,
    PROPOSITION_LEANING_SUBSET,
    ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
)

if proposition_available_attributes:
    fig, _ = create_horizontal_odds_ratio_plot_nominal_grouped(
        proposition_regression_dict,
        split=PROPOSITION_LEANING_SPLIT,
        subset=PROPOSITION_LEANING_SUBSET,
        ylabel=None,
        save_path=build_split_figure_path(
            PROPOSITION_LEANING_SPLIT,
            "distortion_nominal_variables_odds_ratio_by_proposition_leaning.pdf",
        ),
    )
    plt.close(fig)
else:
    print("Skipping nominal odds-ratio by proposition leaning plot due to missing results.")

################################
# NOMINAL ATTRIBUTES - DISTORTION VS WRITER TOLERANCE
################################

OUTCOME_MAP_NOMINAL = {
    "writer_gender": ("gender_different", "gender_different"),
    "writer_race": ("race_different", "race_different"),
    "writer_politicalParty": ("politics_party_different", "politics_party_different"),
}

def match_outcome_distortion_nominal(row):
    labels = OUTCOME_MAP_NOMINAL.get(row.outcome)
    if labels is None:
        return None
    return labels[0]


def create_oddsratio_tolerance_scatterplot_nominal(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    included_splits=None,
    filter_term=None,
    figsize=(9, 6),
    s=40,
    title=None,
    xlabel="Cramer's V for AI vs. Writer Paragraph Distributions.\nHigher = Larger Distribution Shift from AI.",
    ylabel="Average Tolerance Score (0–100).\nHigher = More Accepting of This Distortion.",
    save_path=None,
    annotate_points=True,
):

    fig, ax = plt.subplots(figsize=figsize)

    if included_splits is None:
        included_splits = DEFAULT_PLOT_SPLITS

    included_splits = [
        para_type for para_type in included_splits if para_type in regression_dict
    ]

    if not included_splits:
        raise ValueError("No requested splits are available in regression_dict.")

    default_colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#2E8B57",
    }
    colors = {para_type: default_colors[para_type] for para_type in included_splits}

    if len(included_splits) == 1:
        x_offset = {included_splits[0]: 0}
    else:
        offset_positions = [
            index - (len(included_splits) - 1) / 2
            for index in range(len(included_splits))
        ]
        x_offset = {
            para_type: position * 0.01
            for para_type, position in zip(included_splits, offset_positions)
        }

    for para_type in included_splits:
        if para_type not in regression_dict:
            continue

        regression_df = prepare_nominal_cramers_df(
            regression_dict,
            para_type,
            subset=subset,
        )

        if regression_df.empty:
            continue

        if filter_term:
            regression_df = regression_df[
                regression_df["term"].str.contains(filter_term)
            ]

        regression_df["distortion"] = regression_df.apply(
            match_outcome_distortion_nominal, axis=1
        )

        merged_df = regression_df.merge(
            distortion_tolerance_df,
            on="distortion",
            how="inner",
        )

        if merged_df.empty:
            continue

        x_values = merged_df["cramers_v"] + x_offset[para_type]
        y_values = merged_df["mean"]

        ax.scatter(
            x_values,
            y_values,
            s=s,
            color=colors[para_type],
            edgecolors="black",
            linewidth=1,
            label=para_type.capitalize(),
            zorder=5,
        )

        x_err = [
            merged_df["cramers_v"] - merged_df["cramers_v_ci_low"],
            merged_df["cramers_v_ci_high"] - merged_df["cramers_v"],
        ]

        y_err = [
            merged_df["mean"] - merged_df["mean_ci_low"],
            merged_df["mean_ci_high"] - merged_df["mean"],
        ]

        ax.errorbar(
            x_values,
            y_values,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[para_type],
            zorder=4,
        )

        if annotate_points:
            texts = []
            for _, row in merged_df.iterrows():
                text = ax.annotate(
                    row["outcome"],
                    (row["cramers_v"] + x_offset[para_type], row["mean"]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="square,pad=0.25",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.85,
                    ),
                    zorder=6,
                )
                texts.append(text)

            adjust_text(
                texts,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.7,
                    lw=0.5,
                ),
                force_static=(10, 10),
                force_text=(1, 1),
            )

    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Background tolerance bands
    x_min, x_max = ax.get_xlim()
    band_colors = {
        "0_25": "#cc0000",
        "25_50": "#e06666",
        "50_75": "#93c47d",
        "75_100": "#38761d",
    }

    for lower, color_key in zip(
        [0, 25, 50, 75],
        ["0_25", "25_50", "50_75", "75_100"],
    ):
        ax.add_patch(
            Rectangle(
                (x_min, lower),
                x_max - x_min,
                25,
                facecolor=band_colors[color_key],
                alpha=0.25,
                zorder=0,
            )
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if len(included_splits) > 1:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


for split in ALL_SPLITS:
    if not split_has_required_columns(
        nominal_distribution_dict,
        split,
        NOMINAL_ATTRIBUTES,
        "by_type",
        ["cramers_v", "cramers_v_ci_low", "cramers_v_ci_high"],
    ):
        print(
            f"Skipping nominal tolerance scatter for split '{split}' due to missing Cramer's V results."
        )
        continue

    fig, _ = create_oddsratio_tolerance_scatterplot_nominal(
        nominal_distribution_dict,
        distortion_tolerance_df,
        subset="by_type",
        included_splits=[split],
        save_path=build_split_figure_path(split, "distortion_nominal_variables_tolerance_scatter.pdf"),
    )
    plt.close(fig)
