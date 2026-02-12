import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = "../../results/main_phase_2_distortion/"

################################
# SCALE ATTRIBUTES
################################

SCALE_ATTRIBUTES = [
    "paragraph_formality",
    "paragraph_informativeness",
    "paragraph_originality",
    "paragraph_clarity",
    "paragraph_relevance",
    #
    "writer_knowledge",
    "writer_importance",
    "writer_confidence",
    "writer_stance_polarity",
    #
    "paragraph_hope",
    "paragraph_excitement",
    "paragraph_fear",
    "paragraph_disgust",
    "paragraph_anger",
    #
    "writer_affect_x",
    "writer_affect_y",
    #
    "writer_optimism",
    "writer_community",
    "writer_friendliness",
    "writer_openness",
]

regression_dict = {}

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(SCALE_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in [
            "by_type",
            "by_model",
            "by_input",
        ]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


# Create horizontal AME plot for "by_type" subset
def create_horizontal_ame_plot(
    regression_dict,
    subset="by_type",
    include_unedited=True,
    figsize=(10, 7),
    xlabel="Average Marginal Effect (AME) for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Scale Attributes (Grouped by Type)",
    save_path=None,
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each condition
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }  # Sea green for unedited, crimson for edited
    else:
        colors = {
            "edited": "#000000",
        }  # Black for edited only

    # Define y-position offsets to avoid overlapping points
    y_offset = (
        {"unedited": -0.1, "edited": 0.1} if include_unedited else {"edited": 0}
    )  # No offset if only edited

    # Process both unedited and edited data
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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
    group_boundaries = [3.5, 5.5, 10.5, 14.5]
    for boundary in group_boundaries:
        ax.axhline(boundary, color="gray", linestyle=(0, (5, 5)), linewidth=0.5)

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(SCALE_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(SCALE_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, loc="lower right") if include_unedited else None

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Usage
create_horizontal_ame_plot(
    regression_dict,
    include_unedited=False,
    ylabel=None,
    save_path="../../figures/main_phase_2_distortion/distortion_scale_variables_ame.pdf",
)

################################
# SCALE ATTRIBUTES - DISTORTION VS WRITER TOLERANCE
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


distortion_tolerance_df = pd.read_csv(
    "../../data/main_phase_1/distortion_responses_summary.csv"
)


def create_ame_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=True,
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
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }
        x_offset = {"unedited": -0.005, "edited": 0.005}
    else:
        colors = {"edited": "#000000"}
        x_offset = {"edited": 0}

    # --------------------------------
    # Process conditions
    # --------------------------------
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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

    if include_unedited:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# run
create_ame_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=False,
    save_path="../../figures/main_phase_2_distortion/distortion_scale_variables_tolerance_scatter.pdf",
)


def calculate_ame_tolerance_correlations(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=True,
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

    para_types = ["unedited", "edited"] if include_unedited else ["edited"]

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
    include_unedited=False,
)

print("Correlation Results:", correlation_results)


################################
# ORDINAL ATTRIBUTES
################################

ORDINAL_ATTRIBUTES = [
    "writer_education",
    "writer_english_skills",
    "writer_income",
    "writer_age_binned",
    "writer_english_first",
]

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(ORDINAL_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in ["by_type"]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


# Create horizontal cliffs_delta plot for "by_type" subset
def create_horizontal_cliffs_delta_plot(
    regression_dict,
    subset="by_type",
    include_unedited=True,
    figsize=(10, 2.5),
    xlabel="Cliff's Delta for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Ordinal Attributes",
    save_path=None,
):

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each condition
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }  # Sea green for unedited, crimson for edited
    else:
        colors = {
            "edited": "#000000",
        }  # Black for editedÍ

    # Define y-position offsets to avoid overlapping points
    y_offset = (
        {"unedited": -0.1, "edited": 0.1} if include_unedited else {"edited": 0}
    )  # No offset if only edited

    # Process both unedited and edited data
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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
                regression_df["cliffs_delta"],
                y_positions,
                xerr=[
                    regression_df["cliffs_delta"]
                    - regression_df["cliffs_delta_ci_low"],
                    regression_df["cliffs_delta_ci_high"]
                    - regression_df["cliffs_delta"],
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

    # Set y-tick labels and positions
    y_tick_positions = list(range(len(ORDINAL_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(ORDINAL_ATTRIBUTES)))

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, loc="lower right") if include_unedited else None

    sns.despine(ax=ax)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Usage
create_horizontal_cliffs_delta_plot(
    regression_dict,
    include_unedited=False,
    ylabel=None,
    save_path="../../figures/main_phase_2_distortion/distortion_ordinal_variables_cliffs_delta.pdf",
)

plt.show()


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
    if labels is None or row.cliffs_delta == 0:
        return None
    return labels[0] if row.cliffs_delta > 0 else labels[1]


def create_cliffsdelta_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=True,
    filter_term=None,
    figsize=(9, 6),
    s=40,
    title=None,
    xlabel="Cliff's Delta for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Average Tolerance Score (0–100).\nHigher = More Accepting of This Distortion.",
    save_path=None,
    annotate_points=True,
):

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------
    # Colors + Offsets
    # --------------------------------
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }
        x_offset = {"unedited": -0.01, "edited": 0.01}
    else:
        colors = {"edited": "#000000"}
        x_offset = {"edited": 0}

    # --------------------------------
    # Process conditions
    # --------------------------------
    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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
        x_values = merged_df["cliffs_delta"] + x_offset[para_type]
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
            merged_df["cliffs_delta"] - merged_df["cliffs_delta_ci_low"],
            merged_df["cliffs_delta_ci_high"] - merged_df["cliffs_delta"],
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
                    (row["cliffs_delta"] + x_offset[para_type], row["mean"]),
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

    if include_unedited:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# run
create_cliffsdelta_tolerance_scatterplot(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=False,
    save_path="../../figures/main_phase_2_distortion/distortion_ordinal_variables_tolerance_scatter.pdf",
)

plt.show()


################################
# NOMINAL ATTRIBUTES
################################

NOMINAL_ATTRIBUTES = [
    "writer_politicalParty",
    "writer_race",
    "writer_gender",
]

# Load regression results for each attribute and split
for para in ["unedited", "edited"]:
    regression_dict[para] = {}
    for attr in reversed(NOMINAL_ATTRIBUTES):
        regression_dict[para][attr] = {}
        for split in ["by_type"]:
            regression_dict[para][attr][split] = pd.read_csv(
                os.path.join(
                    RESULTS_DIR,
                    para,
                    f"{attr}_{split}.csv",
                )
            )


def create_horizontal_cramersv_plot_nominal(
    regression_dict,
    subset="by_type",
    include_unedited=True,
    figsize=(10, 3),
    xlabel="Cramér's V for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Nominal Attributes",
    save_path=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }
    else:
        colors = {"edited": "#000000"}

    # Offset
    y_offset = {"unedited": -0.1, "edited": 0.1} if include_unedited else {"edited": 0}

    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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

        y_positions = [
            list(regression_dict[para_type].keys()).index(outcome) + y_offset[para_type]
            for outcome in regression_df["outcome"]
        ]

        ax.errorbar(
            regression_df["cramers_v"],
            y_positions,
            xerr=[
                regression_df["cramers_v"] - regression_df["cramers_v_ci_low"],
                regression_df["cramers_v_ci_high"] - regression_df["cramers_v"],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            elinewidth=1,
            color=colors[para_type],
            label=para_type.capitalize(),
            zorder=4,
        )

    ax.axvline(0, color="black", linestyle=(0, (5, 7)), linewidth=1)

    y_tick_positions = list(range(len(NOMINAL_ATTRIBUTES)))
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(list(reversed(NOMINAL_ATTRIBUTES)))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if include_unedited:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# Usage
create_horizontal_cramersv_plot_nominal(
    regression_dict,
    include_unedited=False,
    ylabel=None,
    save_path="../../figures/main_phase_2_distortion/distortion_nominal_variables_cramers_v.pdf",
)
plt.show()

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
    if labels is None or row.cramers_v == 0:
        return None
    return labels[0] if row.cramers_v > 0 else labels[1]


def create_cramersv_tolerance_scatterplot_nominal(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=True,
    filter_term=None,
    figsize=(9, 6),
    s=40,
    title=None,
    xlabel="Cramér's V for AI vs. Writer Paragraphs.\nHigher = More Distortion from AI.",
    ylabel="Average Tolerance Score (0–100).\nHigher = More Accepting of This Distortion.",
    save_path=None,
    annotate_points=True,
):

    fig, ax = plt.subplots(figsize=figsize)

    if include_unedited:
        colors = {
            "unedited": "#2E8B57",
            "edited": "#DC143C",
        }
        x_offset = {"unedited": -0.01, "edited": 0.01}
    else:
        colors = {"edited": "#000000"}
        x_offset = {"edited": 0}

    for para_type in ["unedited", "edited"] if include_unedited else ["edited"]:
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

    if include_unedited:
        ax.legend(frameon=False, loc="lower right")

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# run
create_cramersv_tolerance_scatterplot_nominal(
    regression_dict,
    distortion_tolerance_df,
    subset="by_type",
    include_unedited=False,
    save_path="../../figures/main_phase_2_distortion/distortion_nominal_variables_tolerance_scatter.pdf",
)
plt.show()
