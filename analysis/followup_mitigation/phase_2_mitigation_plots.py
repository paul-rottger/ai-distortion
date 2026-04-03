import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../results/followup_mitigation_phase_2_distortion"))
FIGURES_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../figures/followup_mitigation_phase_2_distortion"))
DISTORTION_TOLERANCE_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "../../data/main_phase_1/distortion_responses_summary.csv")
)

PARA_TYPES = ["unedited", "edited", "preferred"]
SUBSET = "by_mitigation"
EXCLUDED_ATTRIBUTES = {"writer_affect_x", "writer_affect_y"}

MITIGATION_ORDER = ["reranking", "prompting", "none"]
SUMMARY_MITIGATIONS = ["none", "reranking"]
MITIGATION_TERMS = {
    "none": "mitigation_condition_none",
    "prompting": "mitigation_condition_prompting",
    "reranking": "mitigation_condition_reranking",
}
DISTORTION_TOLERANCE_BAND_COLORS = {
    "0_25": "#cc0000",
    "25_50": "#e06666",
    "50_75": "#93c47d",
    "75_100": "#38761d",
}
ARROW_COLORS = {
    "red": "#cc0000",
    "green": "#38761d",
}
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

################################
# SCALE ATTRIBUTES
################################

def get_scale_attributes(results_dir=RESULTS_DIR, para_type="edited", subset=SUBSET):
    directory = os.path.join(results_dir, para_type)
    suffix = f"_{subset}.csv"
    attributes = [
        filename[: -len(suffix)]
        for filename in os.listdir(directory)
        if filename.endswith(suffix)
        and filename[: -len(suffix)] not in EXCLUDED_ATTRIBUTES
    ]
    return sorted(attributes)


SCALE_ATTRIBUTES = get_scale_attributes()


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


regression_data = load_regression_data(SCALE_ATTRIBUTES)
distortion_tolerance_df = pd.read_csv(DISTORTION_TOLERANCE_PATH)
distortion_tolerance_lookup = distortion_tolerance_df.set_index("distortion")["mean"].to_dict()


# Create horizontal AME plot for "by_mitigation" subset
def create_horizontal_ame_plot(
    regression_data,
    attribute,
    para_types=None,
    figsize=(8, 5),
    xlabel="Average Marginal Effect (AME)\nHigher = More Distortion from AI.",
    ylabel="Mitigation Condition",
    save_path=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "unedited": "#0070C0",
        "edited": "#7030A0",
        "preferred": "#FF0000",
    }
    para_types = para_types or PARA_TYPES
    y_offset = {para_type: 0 for para_type in para_types}

    for para_type in para_types:
        if para_type not in regression_data:
            continue

        if attribute not in regression_data[para_type]:
            continue

        regression_df = regression_data[para_type][attribute]

        # Filter only mitigation rows
        regression_df = regression_df[regression_df["term"].isin(MITIGATION_TERMS.values())]

        # Compute y positions
        y_positions = []
        ame_values = []
        ame_low = []
        ame_high = []

        for mitigation in MITIGATION_ORDER:
            term_name = MITIGATION_TERMS[mitigation]
            row = regression_df[regression_df["term"] == term_name]

            if not row.empty:
                y_positions.append(
                    MITIGATION_ORDER.index(mitigation) + y_offset[para_type]
                )
                ame_values.append(row["ame"].values[0])
                ame_low.append(row["ame_low"].values[0])
                ame_high.append(row["ame_high"].values[0])

        # Plot
        ax.errorbar(
            ame_values,
            y_positions,
            xerr=[
                [ame - low for ame, low in zip(ame_values, ame_low)],
                [high - ame for ame, high in zip(ame_values, ame_high)],
            ],
            fmt="o",
            capsize=3,
            elinewidth=1,
            capthick=1,
            color=colors[para_type],
            label=para_type.capitalize(),
            zorder=4,
        )

    # Reference line at 0
    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    # Y-axis labels
    ax.set_yticks(range(len(MITIGATION_ORDER)))
    ax.set_yticklabels(MITIGATION_ORDER)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_title(attribute)

    if len(para_types) > 1:
        ax.legend(frameon=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

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
        tolerance_mean = distortion_tolerance_lookup.get(distortion_label)
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

os.makedirs(FIGURES_DIR, exist_ok=True)

for para_type in PARA_TYPES:
    para_type_figure_dir = os.path.join(FIGURES_DIR, para_type)
    os.makedirs(para_type_figure_dir, exist_ok=True)

    for attribute in SCALE_ATTRIBUTES:
        fig, ax = create_horizontal_ame_plot(
            regression_data,
            attribute=attribute,
            para_types=[para_type],
            save_path=os.path.join(
                para_type_figure_dir,
                f"{attribute}_ame_by_mitigation.pdf",
            ),
        )
        plt.close(fig)

    fig, ax = create_summary_ame_plot(
        regression_data,
        para_type=para_type,
        scale_attributes=SCALE_ATTRIBUTES,
        save_path=os.path.join(
            para_type_figure_dir,
            "ame_summary_by_mitigation.pdf",
        ),
    )
    if fig is not None:
        plt.close(fig)
