import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../results/followup_mitigation_phase_2_distortion"))
FIGURES_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../figures/followup_mitigation_phase_2"))

PARA_TYPES = ["unedited", "edited"]
SUBSET = "by_mitigation"

MITIGATION_ORDER = ["reranking", "prompting", "none"]
MITIGATION_TERMS = {
    "none": "mitigation_condition_none",
    "prompting": "mitigation_condition_prompting",
    "reranking": "mitigation_condition_reranking",
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


# Create horizontal AME plot for "by_mitigation" subset
def create_horizontal_ame_plot(
    regression_data,
    attribute,
    include_unedited=True,
    figsize=(8, 5),
    xlabel="Average Marginal Effect (AME)\nHigher = More Distortion from AI.",
    ylabel="Mitigation Condition",
    save_path=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    if include_unedited:
        colors = {
            "unedited": "#0070C0",
            "edited": "#7030A0",
        }
        y_offset = {"unedited": -0.1, "edited": 0.1}
    else:
        colors = {"edited": "#000000"}
        y_offset = {"edited": 0}

    para_types = PARA_TYPES if include_unedited else ["edited"]

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

    if include_unedited:
        ax.legend(frameon=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

os.makedirs(FIGURES_DIR, exist_ok=True)

for attribute in SCALE_ATTRIBUTES:
    fig, ax = create_horizontal_ame_plot(
        regression_data,
        attribute=attribute,
        include_unedited=False,
        save_path=os.path.join(FIGURES_DIR, f"{attribute}_ame_by_mitigation.pdf"),
    )
    plt.close(fig)
