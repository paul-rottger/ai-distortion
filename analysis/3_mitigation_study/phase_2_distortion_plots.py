#!/usr/bin/env python3

# =============================================================================
# MITIGATION STUDY - PHASE 2 VISUALIZATION: DISTORTION EFFECTS
#
# Generates phase-2 distortion plots across mitigation conditions.
#
# - Loads distortion analysis results for edited, unedited, and preferred splits.
# - Builds mitigation-comparison figures across scale, ordinal, and nominal outcomes.
# - Saves visual outputs to `figures/followup_mitigation_phase_2_distortion/`.
#
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Package imports
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(BASE_DIR).resolve().parents[1]

sys.path.insert(0, os.path.join(BASE_DIR, "..", "utils_py"))
from demo_paths import get_figures_dir, get_results_dir, get_results_input_dir, parse_demo_mode
from variable_definitions import ORDINAL_VARS as ORDINAL_ATTRIBUTES, NOMINAL_VARS as NOMINAL_ATTRIBUTES
from plotting_utils import get_group_offsets

DEMO_MODE = parse_demo_mode()
RESULTS_DIR = str(get_results_input_dir(REPO_ROOT, "followup_mitigation_phase_2_distortion", demo_mode=DEMO_MODE))
FIGURES_DIR = str(get_figures_dir(REPO_ROOT, "followup_mitigation_phase_2_distortion", demo_mode=DEMO_MODE))
CORRELATION_RESULTS_DIR = str(
    get_results_input_dir(REPO_ROOT, "followup_mitigation_phase_2_distribution", demo_mode=DEMO_MODE)
)
DISTORTION_TOLERANCE_PATH = str(
    get_results_dir(REPO_ROOT, "main_phase_1", "distortion_responses_summary.csv", demo_mode=True)
    if DEMO_MODE
    else REPO_ROOT / "data" / "main_phase_1" / "distortion_responses_summary.csv"
)

# Plot configuration
PARA_TYPES = ["preferred"] if DEMO_MODE else ["unedited", "edited", "preferred"]
SUBSET = "by_mitigation"
EXCLUDED_ATTRIBUTES = {"writer_affect_x", "writer_affect_y"}
GROUPED_SCALE_ATTRIBUTES = [
    "paragraph_formality",
    "paragraph_informativeness",
    "paragraph_originality",
    "paragraph_clarity",
    "paragraph_relevance",
    "writer_knowledge",
    "writer_importance",
    "writer_confidence",
    "writer_stance_polarity",
    "writer_openness",
    "paragraph_hope",
    "paragraph_excitement",
    "paragraph_fear",
    "paragraph_disgust",
    "paragraph_anger",
    "writer_optimism",
    "writer_community",
    "writer_friendliness",
]

MITIGATION_ORDER = ["reranking", "prompting", "none"]
SUMMARY_MITIGATIONS = ["none", "reranking"]
MITIGATION_TERMS = {
    "none": "mitigation_condition_none",
    "prompting": "mitigation_condition_prompting",
    "reranking": "mitigation_condition_reranking",
}
MITIGATION_TERM_ORDER = [MITIGATION_TERMS[mitigation] for mitigation in MITIGATION_ORDER]
MITIGATION_TERM_LABELS = {
    MITIGATION_TERMS["reranking"]: "Reranking",
    MITIGATION_TERMS["prompting"]: "Prompting",
    MITIGATION_TERMS["none"]: "None",
}
MITIGATION_TERM_COLORS = {
    MITIGATION_TERMS["reranking"]: "#0070C0",
    MITIGATION_TERMS["prompting"]: "#B90ED3",
    MITIGATION_TERMS["none"]: "#6E6E6E",
}
DISTORTION_TOLERANCE_BAND_COLORS = {
    "0_25": "#cc0000",
    "25_50": "#e06666",
    "50_75": "#93c47d",
    "75_100": "#38761d",
}
SCALE_GROUP_BOUNDARIES = [2.5, 4.5, 9.5, 14.5]
ARROW_COLORS = {
    "red": "#cc0000",
    "green": "#38761d",
}
CHANGE_VS_STANCE_BASE_MARKER_SIZE = 6
CHANGE_VS_STANCE_MAX_MARKER_SCALE = 3
BEST_FIT_LINE_COLOR = "#4c5c68"
SIDE_EFFECT_CHANGE_GROUP_OVERRIDES = {
    "writer_openness": "liked",
}
SIDE_EFFECT_CHANGE_VS_STANCE_MITIGATIONS = ["reranking"]
OUTCOME_MAP = {
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
SIGNIFICANCE_DETAILS_FILENAME = "mitigation_significance_details.csv"

DISTORTION_TOLERANCE_LOOKUP: dict[str, float] = {}


# =============================================================================
# LOAD DATA
# =============================================================================

def get_scale_attributes(results_dir=RESULTS_DIR, para_type="edited", subset=SUBSET):
    directory = os.path.join(results_dir, para_type)
    suffix = f"_{subset}.csv"
    attributes = []
    for filename in os.listdir(directory):
        if not filename.endswith(suffix):
            continue

        attribute = filename[: -len(suffix)]
        if attribute in EXCLUDED_ATTRIBUTES:
            continue

        file_path = os.path.join(directory, filename)
        columns = pd.read_csv(file_path, nrows=0).columns
        if {"ame", "ame_low", "ame_high"}.issubset(columns):
            attributes.append(attribute)

    return sorted(attributes)

CORRELATION_TARGET_ATTRIBUTE = "writer_stance_polarity"


def load_regression_data(
    scale_attributes,
    results_dir=RESULTS_DIR,
    para_types=PARA_TYPES,
    subset=SUBSET,
):
    regression_data = {para_type: {} for para_type in para_types}
    for para_type in para_types:
        for attr in scale_attributes:
            file_path = os.path.join(results_dir, para_type, f"{attr}_{subset}.csv")
            if os.path.exists(file_path):
                regression_data[para_type][attr] = pd.read_csv(file_path)
    return regression_data


def load_results_by_attribute(
    attributes,
    results_dir=RESULTS_DIR,
    para_types=PARA_TYPES,
    subset=SUBSET,
):
    regression_data = {para_type: {} for para_type in para_types}
    for para_type in para_types:
        for attr in attributes:
            file_path = os.path.join(results_dir, para_type, f"{attr}_{subset}.csv")
            if os.path.exists(file_path):
                regression_data[para_type][attr] = pd.read_csv(file_path)
    return regression_data


def load_correlation_data(
    para_types=PARA_TYPES,
    results_dir=CORRELATION_RESULTS_DIR,
):
    correlation_data = {}
    for para_type in para_types:
        file_path = os.path.join(results_dir, para_type, "scale_attribute_correlations.csv")
        if os.path.exists(file_path):
            correlation_data[para_type] = pd.read_csv(file_path)
    return correlation_data


def load_significance_details(
    para_types=PARA_TYPES,
    results_dir=RESULTS_DIR,
    filename=SIGNIFICANCE_DETAILS_FILENAME,
):
    significance_details = {}
    for para_type in para_types:
        file_path = os.path.join(results_dir, para_type, filename)
        if os.path.exists(file_path):
            significance_details[para_type] = pd.read_csv(file_path)
    return significance_details


def load_distortion_tolerance_lookup(
    distortion_tolerance_path=DISTORTION_TOLERANCE_PATH,
):
    distortion_tolerance_df = pd.read_csv(distortion_tolerance_path)
    return distortion_tolerance_df.set_index("distortion")["mean"].to_dict()


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

def prettify_term_label(term):
    return MITIGATION_TERM_LABELS.get(term, term)


def get_available_attributes(regression_dict, split, attributes, required_columns):
    if split not in regression_dict:
        return []

    available_attributes = []
    for attr in attributes:
        df = regression_dict[split].get(attr)
        if df is None or df.empty or any(column not in df.columns for column in required_columns):
            continue
        available_attributes.append(attr)

    return available_attributes


def get_term_order(terms):
    return [term for term in MITIGATION_TERM_ORDER if term in terms]


def get_available_terms(
    regression_dict,
    split,
    attributes,
    required_columns,
    candidate_attributes=None,
):
    term_sets = []
    target_attributes = candidate_attributes or attributes

    if split not in regression_dict:
        return []

    for attr in target_attributes:
        df = regression_dict[split].get(attr)
        if df is None or df.empty or any(column not in df.columns for column in required_columns):
            return []

        terms = df["term"].dropna().unique().tolist()
        if not terms:
            return []
        term_sets.append(set(terms))

    common_terms = set.intersection(*term_sets)
    return get_term_order(common_terms)


def create_horizontal_grouped_effect_plot(
    regression_dict,
    attributes,
    split,
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
        ["term", estimate_column, lower_column, upper_column],
    )
    if not available_attributes:
        raise ValueError(f"No plottable attributes are available for split '{split}'.")

    included_terms = get_available_terms(
        regression_dict,
        split,
        attributes,
        ["term", estimate_column, lower_column, upper_column],
        candidate_attributes=available_attributes,
    )
    if not included_terms:
        raise ValueError(f"No requested terms are available for split '{split}'.")

    y_offset = get_group_offsets(included_terms)
    attribute_order = list(reversed(available_attributes))
    attribute_positions = {
        attribute: index for index, attribute in enumerate(attribute_order)
    }

    for term in included_terms:
        term_frames = []
        for attr in attribute_order:
            df = regression_dict[split].get(attr)
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
            color=MITIGATION_TERM_COLORS[term],
            label=prettify_term_label(term),
            zorder=4,
        )

    ax.axvline(reference_line, color="black", linestyle=(0, (5, 7)), linewidth=1)

    if group_boundaries:
        for boundary in group_boundaries:
            ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    ax.set_yticks(list(range(len(attribute_order))))
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


def prepare_nominal_regression_df_for_term(regression_dict, para_type, term, attributes=None):
    rows = []
    nominal_attributes = attributes or NOMINAL_ATTRIBUTES

    for attr in reversed(nominal_attributes):
        if attr not in regression_dict.get(para_type, {}):
            continue

        df = regression_dict[para_type][attr].copy()
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
    attributes=None,
):
    comparison_order = []
    seen = set()
    nominal_attributes = attributes or NOMINAL_ATTRIBUTES

    for attr in reversed(nominal_attributes):
        for term in included_terms:
            if attr not in regression_dict.get(para_type, {}):
                continue

            df = regression_dict[para_type][attr]
            df = df[df["term"] == term]
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


def create_horizontal_odds_ratio_plot_nominal_grouped(
    regression_dict,
    split,
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
        ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
    )
    if not available_attributes:
        raise ValueError(f"No plottable attributes are available for split '{split}'.")

    included_terms = get_available_terms(
        regression_dict,
        split,
        NOMINAL_ATTRIBUTES,
        ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
        candidate_attributes=available_attributes,
    )
    if not included_terms:
        raise ValueError(f"No requested terms are available for split '{split}'.")

    comparison_order = get_nominal_comparison_order_for_terms(
        regression_dict,
        split,
        included_terms,
        attributes=available_attributes,
    )
    group_boundaries = get_nominal_group_boundaries(comparison_order)
    comparison_positions = {label: index for index, label in enumerate(comparison_order)}
    y_offset = get_group_offsets(included_terms)

    for term in included_terms:
        regression_df = prepare_nominal_regression_df_for_term(
            regression_dict,
            split,
            term,
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
            color=MITIGATION_TERM_COLORS[term],
            label=prettify_term_label(term),
            zorder=4,
        )

    ax.axvline(1, color="black", linestyle=(0, (5, 7)), linewidth=1)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    ax.set_yticks(list(range(len(comparison_order))))
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


def create_summary_rows(
    regression_data,
    para_type,
    scale_attributes,
    mitigations=SUMMARY_MITIGATIONS,
):
    summary_rows = []

    for attribute in scale_attributes:
        if attribute not in regression_data.get(para_type, {}):
            continue

        regression_df = regression_data[para_type][attribute]

        for mitigation in mitigations:
            term_name = MITIGATION_TERMS[mitigation]
            row = regression_df[regression_df["term"] == term_name]

            if row.empty:
                continue

            summary_rows.append(
                {
                    "attribute": attribute,
                    "attribute_label": attribute.replace("_", " "),
                    "mitigation": mitigation,
                    "ame": row["ame"].values[0],
                    "ame_low": row["ame_low"].values[0],
                    "ame_high": row["ame_high"].values[0],
                }
            )

    return pd.DataFrame(summary_rows)


def get_nonsignificant_summary_attributes(summary_df):
    ci_df = summary_df.pivot(index="attribute", columns="mitigation", values="ame")
    ci_low_df = summary_df.pivot(index="attribute", columns="mitigation", values="ame_low")
    ci_high_df = summary_df.pivot(index="attribute", columns="mitigation", values="ame_high")

    required_mitigations = {"none", "reranking"}
    if not required_mitigations.issubset(ci_df.columns):
        return set()

    overlapping_ci_mask = (
        (ci_low_df["none"] <= ci_high_df["reranking"])
        & (ci_low_df["reranking"] <= ci_high_df["none"])
    )
    return set(overlapping_ci_mask[overlapping_ci_mask].index)


def calculate_percent_change(none_ame, reranking_ame):
    if pd.isna(none_ame) or pd.isna(reranking_ame) or none_ame == 0:
        return None

    return ((reranking_ame - none_ame) / abs(none_ame)) * 100


def format_p_value(p_value):
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}"


def get_significance_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def get_plot_correlations(plot_df, x_column="percent_change"):
    valid_df = plot_df[[x_column, "stance_correlation"]].dropna()

    if len(valid_df) < 3:
        return None

    pearson_r, pearson_p = pearsonr(
        valid_df[x_column],
        valid_df["stance_correlation"],
    )
    spearman_r, spearman_p = spearmanr(
        valid_df[x_column],
        valid_df["stance_correlation"],
    )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def create_change_correlation_rows(
    regression_data,
    correlation_data,
    para_type,
    scale_attributes,
    correlation_target=CORRELATION_TARGET_ATTRIBUTE,
):
    summary_df = create_summary_rows(
        regression_data,
        para_type,
        scale_attributes,
        mitigations=SUMMARY_MITIGATIONS,
    )
    if summary_df.empty or para_type not in correlation_data:
        return pd.DataFrame()

    nonsignificant_attributes = get_nonsignificant_summary_attributes(summary_df)
    ame_df = summary_df.pivot(index="attribute", columns="mitigation", values="ame")

    if not {"none", "reranking"}.issubset(ame_df.columns):
        return pd.DataFrame()

    row_backgrounds = get_summary_row_backgrounds(
        summary_df,
        ame_df.index.tolist(),
        nonsignificant_attributes,
    )

    stance_correlation_df = correlation_data[para_type]
    stance_correlation_df = stance_correlation_df[
        stance_correlation_df["attribute_y"] == correlation_target
    ][["attribute_x", "correlation", "p_value"]].rename(
        columns={"attribute_x": "attribute", "correlation": "stance_correlation"}
    )

    rows = []
    for attribute in scale_attributes:
        if attribute in nonsignificant_attributes:
            continue
        if attribute not in ame_df.index:
            continue

        percent_change = calculate_percent_change(
            ame_df.at[attribute, "none"],
            ame_df.at[attribute, "reranking"],
        )
        if percent_change is None:
            continue

        correlation_row = stance_correlation_df[
            stance_correlation_df["attribute"] == attribute
        ]
        if correlation_row.empty or pd.isna(correlation_row["stance_correlation"].iloc[0]):
            continue

        stance_correlation = correlation_row["stance_correlation"].iloc[0]
        if attribute == "writer_openness":
            percent_change *= -1
            stance_correlation *= -1

        rows.append(
            {
                "attribute": attribute,
                "attribute_label": attribute.replace("_", " "),
                "percent_change": percent_change,
                "stance_correlation": stance_correlation,
                "correlation_p_value": correlation_row["p_value"].iloc[0],
                "none_ame": ame_df.at[attribute, "none"],
                "reranking_ame": ame_df.at[attribute, "reranking"],
                "change_group": get_change_group(
                    row_backgrounds.get(attribute),
                    ame_df.at[attribute, "none"],
                    ame_df.at[attribute, "reranking"],
                ),
            }
        )

    return pd.DataFrame(rows)


def get_distortion_label(attribute, ame_value):
    labels = OUTCOME_MAP.get(attribute)
    if labels is None or pd.isna(ame_value) or ame_value == 0:
        return None
    return labels[0] if ame_value > 0 else labels[1]


def get_tolerance_band_color(tolerance_mean):
    if pd.isna(tolerance_mean):
        return None
    if tolerance_mean < 25:
        return DISTORTION_TOLERANCE_BAND_COLORS["0_25"]
    if tolerance_mean < 50:
        return DISTORTION_TOLERANCE_BAND_COLORS["25_50"]
    if tolerance_mean < 75:
        return DISTORTION_TOLERANCE_BAND_COLORS["50_75"]
    return DISTORTION_TOLERANCE_BAND_COLORS["75_100"]


def lighten_color(color, amount=0.45):
    red, green, blue = to_rgb(color)
    return (
        red + (1 - red) * amount,
        green + (1 - green) * amount,
        blue + (1 - blue) * amount,
    )


def get_arrow_color(background_color):
    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["50_75"],
        DISTORTION_TOLERANCE_BAND_COLORS["75_100"],
    }:
        return ARROW_COLORS["red"]
    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["0_25"],
        DISTORTION_TOLERANCE_BAND_COLORS["25_50"],
    }:
        return ARROW_COLORS["green"]
    return None


def get_summary_row_backgrounds(summary_df, attribute_order, nonsignificant_attributes):
    none_df = (
        summary_df[summary_df["mitigation"] == "none"]
        .set_index("attribute")
        .reindex(attribute_order)
    )

    row_backgrounds = {}
    for attribute in attribute_order:
        if attribute in nonsignificant_attributes:
            row_backgrounds[attribute] = "#c5c5c5"
            continue

        if attribute not in none_df.index:
            row_backgrounds[attribute] = None
            continue

        distortion_label = get_distortion_label(attribute, none_df.at[attribute, "ame"])
        tolerance_mean = DISTORTION_TOLERANCE_LOOKUP.get(distortion_label)
        row_backgrounds[attribute] = get_tolerance_band_color(tolerance_mean)

    return row_backgrounds


def get_attribute_group(attribute, row_backgrounds, nonsignificant_attributes):
    if attribute == "writer_stance_polarity":
        return "stance"
    if attribute in nonsignificant_attributes:
        return "insignificant"

    background_color = row_backgrounds.get(attribute)
    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["50_75"],
        DISTORTION_TOLERANCE_BAND_COLORS["75_100"],
    }:
        return "liked"
    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["0_25"],
        DISTORTION_TOLERANCE_BAND_COLORS["25_50"],
    }:
        return "disliked"
    return "insignificant"


def get_change_group(background_color, none_ame, reranking_ame):
    if pd.isna(none_ame) or pd.isna(reranking_ame) or background_color is None:
        return None

    none_distortion = abs(none_ame)
    reranking_distortion = abs(reranking_ame)

    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["0_25"],
        DISTORTION_TOLERANCE_BAND_COLORS["25_50"],
    }:
        return "liked" if reranking_distortion < none_distortion else "disliked"

    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["50_75"],
        DISTORTION_TOLERANCE_BAND_COLORS["75_100"],
    }:
        return "liked" if reranking_distortion > none_distortion else "disliked"

    return None


def get_grouped_attribute_order(attribute_diff_df, row_backgrounds, nonsignificant_attributes):
    group_order = ["stance", "disliked", "liked", "insignificant"]
    grouped_attributes = {group: [] for group in group_order}

    for attribute, row in attribute_diff_df.iterrows():
        group = get_attribute_group(attribute, row_backgrounds, nonsignificant_attributes)
        grouped_attributes[group].append((attribute, row["none_ame"]))

    ordered_attributes = []
    present_groups = []
    for group in group_order:
        group_attributes = sorted(
            grouped_attributes[group],
            key=lambda item: item[1],
            reverse=True,
        )
        if not group_attributes:
            continue
        present_groups.append(group)
        ordered_attributes.extend(attribute for attribute, _ in group_attributes)

    return ordered_attributes, present_groups


def get_attribute_positions(attribute_order, row_backgrounds, nonsignificant_attributes):
    group_spacing = 0.9
    positions = {}
    y_position = 0.0
    previous_group = None

    for attribute in attribute_order:
        group = get_attribute_group(attribute, row_backgrounds, nonsignificant_attributes)
        if previous_group is not None and group != previous_group:
            y_position += group_spacing
        positions[attribute] = y_position
        y_position += 1.0
        previous_group = group

    return positions


def add_significance_arrow(ax, none_ame, reranking_ame, y_position, background_color):
    arrow_color = get_arrow_color(background_color)
    if arrow_color is None:
        return

    arrow_y = y_position - 0.3
    ax.annotate(
        "",
        xy=(reranking_ame, arrow_y),
        xytext=(none_ame, arrow_y),
        arrowprops={
            "arrowstyle": "->",
            "color": arrow_color,
            "linewidth": 1.4,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=3.5,
    )


def create_summary_ame_plot(
    regression_data,
    para_type,
    scale_attributes,
    mitigations=SUMMARY_MITIGATIONS,
    xlabel="Average Marginal Effect (AME)\nHigher = More Distortion from AI.",
    ylabel="Attribute",
    save_path=None,
):
    summary_df = create_summary_rows(
        regression_data,
        para_type,
        scale_attributes,
        mitigations=mitigations,
    )

    if summary_df.empty:
        return None, None

    fig_height = max(6, len(summary_df["attribute"].unique()) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    colors = {
        "none": "#6E6E6E",
        "reranking": "#0070C0",
    }
    nonsignificant_attributes = get_nonsignificant_summary_attributes(summary_df)
    attribute_diff_df = (
        summary_df.pivot(index="attribute", columns="mitigation", values="ame")
        .reindex(columns=mitigations)
    )
    attribute_diff_df["none_ame"] = attribute_diff_df["none"]
    row_backgrounds = get_summary_row_backgrounds(
        summary_df,
        attribute_diff_df.index.tolist(),
        nonsignificant_attributes,
    )
    attribute_order, _ = get_grouped_attribute_order(
        attribute_diff_df,
        row_backgrounds,
        nonsignificant_attributes,
    )
    attribute_positions = get_attribute_positions(
        attribute_order,
        row_backgrounds,
        nonsignificant_attributes,
    )

    for attribute, position in attribute_positions.items():
        background_color = row_backgrounds.get(attribute)
        if background_color is not None:
            ax.axhspan(
                position - 0.5,
                position + 0.5,
                facecolor=background_color,
                edgecolor="none",
                alpha=0.25,
                zorder=0,
            )

    if len(mitigations) == 1:
        y_offset = {mitigations[0]: 0}
    else:
        y_offset = {mitigation: 0 for mitigation in mitigations}

    display_colors = {
        mitigation: lighten_color(colors[mitigation]) for mitigation in mitigations
    }

    for mitigation in mitigations:
        mitigation_df = summary_df[summary_df["mitigation"] == mitigation]

        if mitigation_df.empty:
            continue

        y_positions = [
            attribute_positions[attribute] + y_offset[mitigation]
            for attribute in mitigation_df["attribute"]
        ]

        point_alphas = mitigation_df["attribute"].map(
            lambda attribute: 1.0
        )

        for (_, row), y_position, point_alpha in zip(
            mitigation_df.iterrows(),
            y_positions,
            point_alphas,
        ):
            xerr = [
                [row["ame"] - row["ame_low"]],
                [row["ame_high"] - row["ame"]],
            ]
            line_color = (
                display_colors[mitigation]
                if row["attribute"] in nonsignificant_attributes
                else colors[mitigation]
            )

            ax.errorbar(
                [row["ame"]],
                [y_position],
                xerr=xerr,
                fmt="o",
                capsize=3,
                elinewidth=1,
                capthick=1,
                color=line_color,
                alpha=point_alpha,
                zorder=4,
            )

    if {"none", "reranking"}.issubset(set(mitigations)):
        none_df = (
            summary_df[summary_df["mitigation"] == "none"]
            .set_index("attribute")
            .reindex(attribute_order)
        )
        reranking_df = (
            summary_df[summary_df["mitigation"] == "reranking"]
            .set_index("attribute")
            .reindex(attribute_order)
        )

        for attribute in attribute_order:
            if attribute in nonsignificant_attributes:
                continue
            if attribute not in none_df.index or attribute not in reranking_df.index:
                continue
            if pd.isna(none_df.at[attribute, "ame"]) or pd.isna(reranking_df.at[attribute, "ame"]):
                continue

            add_significance_arrow(
                ax,
                none_df.at[attribute, "ame"],
                reranking_df.at[attribute, "ame"],
                attribute_positions[attribute],
                row_backgrounds.get(attribute),
            )

    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)
    ax.set_yticks([attribute_positions[attribute] for attribute in attribute_order])
    ax.set_yticklabels([attribute.replace("_", " ") for attribute in attribute_order])
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_title(para_type.capitalize())
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[mitigation],
            markersize=5,
            label=mitigation.capitalize(),
        )
        for mitigation in mitigations
    ]
    ax.legend(handles=legend_handles, frameon=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def create_change_vs_stance_correlation_plot(
    regression_data,
    correlation_data,
    para_type,
    scale_attributes,
    xlabel="Percent Change in AME from None to Reranking (%)",
    ylabel="Correlation with writer stance polarity",
    save_path=None,
):
    plot_df = create_change_correlation_rows(
        regression_data,
        correlation_data,
        para_type,
        scale_attributes,
    )

    if plot_df.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 6))
    point_colors = plot_df["change_group"].map(
        {
            "liked": ARROW_COLORS["green"],
            "disliked": ARROW_COLORS["red"],
        }
    ).fillna("#7a7a7a")
    correlation_stats = get_plot_correlations(plot_df)

    ax.scatter(
        plot_df["percent_change"],
        plot_df["stance_correlation"],
        s=42,
        c=point_colors,
        alpha=0.9,
        zorder=3,
    )

    texts = []
    for _, row in plot_df.iterrows():
        texts.append(
            ax.text(
                row["percent_change"],
                row["stance_correlation"],
                row["attribute_label"],
                fontsize=8,
                ha="left",
                va="bottom",
            )
        )

    adjust_text(
        texts,
        ax=ax,
        only_move={"points": "y", "text": "xy"},
        arrowprops={"arrowstyle": "-", "color": "#7a7a7a", "lw": 0.6},
    )

    ax.axhline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
    ax.axvline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(para_type.capitalize())

    if correlation_stats is not None:
        correlation_text = (
            f"Pearson r = {correlation_stats['pearson_r']:.2f} "
            f"(p {format_p_value(correlation_stats['pearson_p'])})\n"
            f"Spearman rho = {correlation_stats['spearman_r']:.2f} "
            f"(p {format_p_value(correlation_stats['spearman_p'])})"
        )
        ax.text(
            0.02,
            0.02,
            correlation_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "#f0f0f0", "edgecolor": "none", "alpha": 1.0},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["green"],
            markersize=6,
            label="Favorable change",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["red"],
            markersize=6,
            label="Unfavorable change",
        ),
    ]
    legend = ax.legend(handles=legend_handles, frameon=True)
    legend.get_frame().set_facecolor("#f0f0f0")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("none")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def get_linear_fit_points(x_values, y_values, num_points=200):
    pair_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

    if len(pair_df) < 2 or pair_df["x"].nunique() < 2:
        return None

    slope, intercept = np.polyfit(pair_df["x"], pair_df["y"], 1)
    x_fit = np.linspace(pair_df["x"].min(), pair_df["x"].max(), num_points)
    y_fit = slope * x_fit + intercept

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def get_fraction_toward_writer_ame(ame_value, writer_ame_value):
    if pd.isna(ame_value) or pd.isna(writer_ame_value) or writer_ame_value == 0:
        return 0.0

    if np.sign(ame_value) != np.sign(writer_ame_value):
        return 0.0

    return float(np.clip(abs(ame_value) / abs(writer_ame_value), 0.0, 1.0))


def get_change_vs_stance_marker_size(
    is_significant,
    ame_value,
    writer_ame_value,
    base_marker_size=CHANGE_VS_STANCE_BASE_MARKER_SIZE,
    max_marker_scale=CHANGE_VS_STANCE_MAX_MARKER_SCALE,
):
    if not is_significant:
        return base_marker_size

    fraction_toward_writer = get_fraction_toward_writer_ame(ame_value, writer_ame_value)
    return base_marker_size * (
        1 + fraction_toward_writer * (max_marker_scale - 1)
    )


def get_side_effect_distortion_label(attribute, ame_value):
    labels = OUTCOME_MAP.get(attribute)
    if labels is None or pd.isna(ame_value) or ame_value == 0:
        return None
    return labels[1] if ame_value > 0 else labels[0]


def get_side_effect_change_group(attribute, background_color, effect_value):
    override_group = SIDE_EFFECT_CHANGE_GROUP_OVERRIDES.get(attribute)
    if override_group is not None:
        return override_group

    if pd.isna(effect_value) or background_color is None:
        return None

    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["0_25"],
        DISTORTION_TOLERANCE_BAND_COLORS["25_50"],
    }:
        if effect_value < 0:
            return "liked"
        if effect_value > 0:
            return "disliked"
        return None

    if background_color in {
        DISTORTION_TOLERANCE_BAND_COLORS["50_75"],
        DISTORTION_TOLERANCE_BAND_COLORS["75_100"],
    }:
        if effect_value > 0:
            return "liked"
        if effect_value < 0:
            return "disliked"
        return None

    return None


def build_side_effect_correlation_rows(
    significance_details_df,
    correlation_df,
    mitigation,
    correlation_target=CORRELATION_TARGET_ATTRIBUTE,
    significant_only=True,
):
    if significance_details_df.empty or correlation_df.empty:
        return pd.DataFrame()

    mitigation_df = significance_details_df[
        significance_details_df["mitigation"] == mitigation
    ].copy()
    mitigation_df = mitigation_df[
        ~mitigation_df["attribute"].isin(EXCLUDED_ATTRIBUTES)
    ]

    if significant_only:
        mitigation_df = mitigation_df[mitigation_df["significant"]]

    if mitigation_df.empty:
        return pd.DataFrame()

    stance_correlation_df = correlation_df[
        correlation_df["attribute_y"] == correlation_target
    ][["attribute_x", "correlation", "p_value"]].rename(
        columns={
            "attribute_x": "attribute",
            "correlation": "stance_correlation",
            "p_value": "correlation_p_value",
        }
    )

    plot_df = mitigation_df.merge(stance_correlation_df, on="attribute", how="inner")

    if plot_df.empty:
        return pd.DataFrame()

    plot_df["attribute_label"] = plot_df["attribute"].str.replace("_", " ")
    plot_df["background_color"] = plot_df.apply(
        lambda row: get_tolerance_band_color(
            DISTORTION_TOLERANCE_LOOKUP.get(
                get_side_effect_distortion_label(row["attribute"], row["writer_ame"])
            )
        ),
        axis=1,
    )
    plot_df["change_group"] = plot_df.apply(
        lambda row: get_side_effect_change_group(
            row["attribute"],
            row["background_color"],
            row["ame"],
        ),
        axis=1,
    )

    return plot_df.dropna(subset=["ame", "stance_correlation"])


def build_mitigation_reduction_rows(significance_details_df, mitigation):
    if significance_details_df.empty:
        return pd.DataFrame()

    plot_df = significance_details_df[
        significance_details_df["mitigation"] == mitigation
    ].copy()
    plot_df = plot_df[~plot_df["attribute"].isin(EXCLUDED_ATTRIBUTES)].copy()
    plot_df = plot_df[plot_df["significant"]].copy()

    if plot_df.empty:
        return pd.DataFrame()

    plot_df["reduction_fraction"] = plot_df.apply(
        lambda row: get_fraction_toward_writer_ame(row["ame"], row["writer_ame"]),
        axis=1,
    )
    plot_df = plot_df[plot_df["reduction_fraction"] > 0].copy()

    if plot_df.empty:
        return pd.DataFrame()

    plot_df["background_color"] = plot_df.apply(
        lambda row: get_tolerance_band_color(
            DISTORTION_TOLERANCE_LOOKUP.get(
                get_side_effect_distortion_label(row["attribute"], row["writer_ame"])
            )
        ),
        axis=1,
    )
    plot_df["change_group"] = plot_df.apply(
        lambda row: get_side_effect_change_group(
            row["attribute"],
            row["background_color"],
            row["ame"],
        ),
        axis=1,
    )
    plot_df = plot_df[plot_df["change_group"].isin(["liked", "disliked"])].copy()

    if plot_df.empty:
        return pd.DataFrame()

    plot_df["reduction_percent"] = -plot_df["reduction_fraction"] * 100
    plot_df["attribute_label"] = plot_df.apply(
        lambda row: row["attribute"].replace("_", " ") + get_significance_stars(row["p_value"]),
        axis=1,
    )
    plot_df = plot_df.sort_values(
        by=["reduction_fraction", "attribute_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return plot_df


def create_mitigation_reduction_plot(
    significance_details_df,
    mitigation,
    xlabel="Reduction in distortion toward writer (%)",
    save_path=None,
):
    plot_df = build_mitigation_reduction_rows(significance_details_df, mitigation)

    if plot_df.empty:
        return None, None

    fig_height = max(4.0, 0.38 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y_positions = np.arange(len(plot_df))
    arrow_colors = {
        "liked": ARROW_COLORS["green"],
        "disliked": ARROW_COLORS["red"],
    }

    for y_position, row in zip(y_positions, plot_df.itertuples(index=False)):
        point_color = arrow_colors[row.change_group]
        ax.annotate(
            "",
            xy=(row.reduction_percent, y_position),
            xytext=(0, y_position),
            arrowprops={
                "arrowstyle": "-|>",
                "color": point_color,
                "lw": 1.8,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )
        ax.plot(
            row.reduction_percent,
            y_position,
            linestyle="",
            marker="o",
            markersize=4.5,
            color=point_color,
            markeredgewidth=0,
            zorder=3,
        )
        ax.text(
            row.reduction_percent - 1.0,
            y_position,
            f"{row.reduction_percent:.1f}%",
            va="center",
            ha="right",
            fontsize=9,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["attribute_label"])
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelright=True, labelleft=False)
    ax.invert_yaxis()
    ax.set_xlim(min(-100, plot_df["reduction_percent"].min() - 8), 0)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{mitigation.capitalize()} distortion reduction")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["green"],
            markersize=6,
            label="Favorable change",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["red"],
            markersize=6,
            label="Unfavorable change",
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower left")

    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


def create_change_vs_stance_correlation_plot_from_significance_details(
    significance_details_df,
    correlation_df,
    mitigation,
    correlation_target=CORRELATION_TARGET_ATTRIBUTE,
    significant_only=False,
    xlabel="Average marginal effect",
    ylabel="Correlation with writer stance polarity",
    save_path=None,
):
    plot_df = build_side_effect_correlation_rows(
        significance_details_df,
        correlation_df,
        mitigation=mitigation,
        correlation_target=correlation_target,
        significant_only=significant_only,
    )

    if plot_df.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 6))
    point_colors = plot_df.apply(
        lambda row: "#7a7a7a"
        if not row["significant"]
        else {
            "liked": ARROW_COLORS["green"],
            "disliked": ARROW_COLORS["red"],
        }.get(row["change_group"], "#7a7a7a"),
        axis=1,
    )
    point_sizes = plot_df.apply(
        lambda row: get_change_vs_stance_marker_size(
            row["significant"],
            row["ame"],
            row["writer_ame"],
        ),
        axis=1,
    )
    correlation_stats = get_plot_correlations(plot_df, x_column="ame")
    linear_fit = get_linear_fit_points(plot_df["ame"], plot_df["stance_correlation"])

    if linear_fit is not None:
        ax.plot(
            linear_fit["x_fit"],
            linear_fit["y_fit"],
            color=BEST_FIT_LINE_COLOR,
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            zorder=2,
        )

    for row, point_color, point_size in zip(
        plot_df.itertuples(index=False),
        point_colors,
        point_sizes,
    ):
        ax.plot(
            row.ame,
            row.stance_correlation,
            linestyle="",
            marker="o",
            markersize=point_size,
            color=point_color,
            markeredgewidth=0,
            alpha=0.9,
            zorder=3,
        )

    texts = []
    for _, row in plot_df.iterrows():
        texts.append(
            ax.text(
                row["ame"],
                row["stance_correlation"],
                row["attribute_label"],
                fontsize=8,
                ha="left",
                va="bottom",
            )
        )

    adjust_text(
        texts,
        ax=ax,
        force_static=(0.3, 0.5),
        force_explode=(0.3, 0.8),
        explode_radius=200,
        max_move=(20, 20),
        expand=(1.2, 1.4),
        only_move={"static": "y", "text": "xy", "explode": "xy", "pull": "xy"},
        arrowprops={"arrowstyle": "-", "color": "#7a7a7a", "lw": 0.6},
    )

    ax.axhline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
    ax.axvline(0, color="#9b9b9b", linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(mitigation.capitalize())

    if correlation_stats is not None:
        correlation_text = (
            f"Pearson r = {correlation_stats['pearson_r']:.2f} "
            f"(p {format_p_value(correlation_stats['pearson_p'])})\n"
            f"Spearman rho = {correlation_stats['spearman_r']:.2f} "
            f"(p {format_p_value(correlation_stats['spearman_p'])})"
        )
        ax.text(
            0.02,
            0.02,
            correlation_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            color=BEST_FIT_LINE_COLOR,
            linewidth=1.5,
            label="Linear fit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["green"],
            markersize=6,
            label="Favorable change",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ARROW_COLORS["red"],
            markersize=6,
            label="Unfavorable change",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#7a7a7a",
            markersize=6,
            label="Insignificant change",
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig, ax


def main():
    global DISTORTION_TOLERANCE_LOOKUP

    scale_attributes = get_scale_attributes()
    regression_data = load_regression_data(scale_attributes)
    grouped_scale_data = load_results_by_attribute(GROUPED_SCALE_ATTRIBUTES)
    ordinal_regression_data = load_results_by_attribute(ORDINAL_ATTRIBUTES)
    nominal_regression_data = load_results_by_attribute(NOMINAL_ATTRIBUTES)
    correlation_data = load_correlation_data()
    significance_details_data = load_significance_details()
    DISTORTION_TOLERANCE_LOOKUP = load_distortion_tolerance_lookup()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for para_type in PARA_TYPES:
        para_type_figure_dir = os.path.join(FIGURES_DIR, para_type)
        os.makedirs(para_type_figure_dir, exist_ok=True)

        available_scale_attributes = get_available_attributes(
            grouped_scale_data,
            para_type,
            GROUPED_SCALE_ATTRIBUTES,
            ["term", "ame", "ame_low", "ame_high"],
        )
        if available_scale_attributes:
            fig, _ = create_horizontal_grouped_effect_plot(
                grouped_scale_data,
                attributes=GROUPED_SCALE_ATTRIBUTES,
                split=para_type,
                estimate_column="ame",
                lower_column="ame_low",
                upper_column="ame_high",
                figsize=(8, 10),
                xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nLarger = More Distortion from AI.",
                ylabel=None,
                reference_line=0,
                save_path=build_split_figure_path(
                    para_type,
                    "distortion_scale_variables_ame_by_mitigation.pdf",
                ),
                group_boundaries=SCALE_GROUP_BOUNDARIES,
            )
            plt.close(fig)

        available_ordinal_attributes = get_available_attributes(
            ordinal_regression_data,
            para_type,
            ORDINAL_ATTRIBUTES,
            ["term", "odds_ratio", "or_low", "or_high"],
        )
        if available_ordinal_attributes:
            fig, _ = create_horizontal_grouped_effect_plot(
                ordinal_regression_data,
                attributes=ORDINAL_ATTRIBUTES,
                split=para_type,
                estimate_column="odds_ratio",
                lower_column="or_low",
                upper_column="or_high",
                figsize=(8, 4),
                xlabel="Odds Ratio for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
                ylabel=None,
                reference_line=1,
                save_path=build_split_figure_path(
                    para_type,
                    "distortion_ordinal_variables_odds_ratio_by_mitigation.pdf",
                ),
            )
            plt.close(fig)

        available_nominal_attributes = get_available_attributes(
            nominal_regression_data,
            para_type,
            NOMINAL_ATTRIBUTES,
            ["term", "odds_ratio", "or_low", "or_high", "target_level", "reference_level"],
        )
        if available_nominal_attributes:
            fig, _ = create_horizontal_odds_ratio_plot_nominal_grouped(
                nominal_regression_data,
                split=para_type,
                ylabel=None,
                save_path=build_split_figure_path(
                    para_type,
                    "distortion_nominal_variables_odds_ratio_by_mitigation.pdf",
                ),
            )
            plt.close(fig)

        fig, _ = create_summary_ame_plot(
            regression_data,
            para_type=para_type,
            scale_attributes=scale_attributes,
            save_path=os.path.join(
                para_type_figure_dir,
                "ame_summary_by_mitigation.pdf",
            ),
        )
        if fig is not None:
            plt.close(fig)

        fig, _ = create_change_vs_stance_correlation_plot(
            regression_data,
            correlation_data,
            para_type=para_type,
            scale_attributes=scale_attributes,
            save_path=os.path.join(
                para_type_figure_dir,
                "ame_change_vs_stance_correlation.pdf",
            ),
        )
        if fig is not None:
            plt.close(fig)

        if para_type in significance_details_data and para_type in correlation_data:
            fig, _ = create_mitigation_reduction_plot(
                significance_details_data[para_type],
                mitigation="reranking",
                save_path=os.path.join(
                    para_type_figure_dir,
                    "reranking_distortion_reduction_vs_writer.pdf",
                ),
            )
            if fig is not None:
                plt.close(fig)

            for mitigation in SIDE_EFFECT_CHANGE_VS_STANCE_MITIGATIONS:
                fig, _ = create_change_vs_stance_correlation_plot_from_significance_details(
                    significance_details_data[para_type],
                    correlation_data[para_type],
                    mitigation=mitigation,
                    save_path=os.path.join(
                        para_type_figure_dir,
                        f"ame_change_vs_stance_correlation_{mitigation}_vs_writer.pdf",
                    ),
                )
                if fig is not None:
                    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
