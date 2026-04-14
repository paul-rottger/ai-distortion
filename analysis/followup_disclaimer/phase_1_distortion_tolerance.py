import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# set up plot format: latex fonts
from matplotlib import rc, font_manager

font_files = font_manager.findSystemFonts(
    fontpaths="/Users/paul/Library/Fonts", fontext="ttf"
)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
rc("text", usetex=False)
plt.rcParams.update({"text.color": "black"})
plt.rcParams.update({"font.size": 12})


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "../../data/followup_disclaimer_phase_1")
)
FIGURES_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "../../figures/followup_disclaimer_phase_1")
)


# load data
distortion_df = pd.read_csv(os.path.join(DATA_DIR, "distortion_responses.csv"))


def create_summary_with_bootstrap_ci(
    df: pd.DataFrame,
    id_col: str = "writer_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    attributes = [col for col in df.columns if col != id_col]

    alpha = 1 - ci
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    results = []

    for col in attributes:
        values = df[col].dropna().to_numpy()
        n = len(values)

        if n == 0:
            continue

        # Point estimates
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values, ddof=1)

        # Bootstrap
        boot_means = np.empty(n_boot)
        boot_medians = np.empty(n_boot)

        for i in range(n_boot):
            sample = rng.choice(values, size=n, replace=True)
            boot_means[i] = np.mean(sample)
            boot_medians[i] = np.median(sample)

        mean_ci_low, mean_ci_high = np.quantile(boot_means, [lower_q, upper_q])
        median_ci_low, median_ci_high = np.quantile(boot_medians, [lower_q, upper_q])

        results.append(
            {
                "distortion": col,
                "mean": mean_val,
                "mean_ci_low": mean_ci_low,
                "mean_ci_high": mean_ci_high,
                "median": median_val,
                "median_ci_low": median_ci_low,
                "median_ci_high": median_ci_high,
                "std": std_val,
            }
        )

    if save_path is not None:
        pd.DataFrame(results).to_csv(save_path, index=False)

    return pd.DataFrame(results)


summary_df = create_summary_with_bootstrap_ci(
    distortion_df,
    id_col="writer_id",
    n_boot=1000,
    ci=0.95,
    random_state=42,
    save_path=os.path.join(DATA_DIR, "distortion_responses_summary.csv"),
)

distortion_label_mapping = {
    "extreme_more": "[...] my opinion on an issue seem more extreme [...]",
    "moderate_more": "[...] my opinion on an issue seem more moderate [...]",
    "confidence_more": "[...] me seem more confident in my opinion [...]",
    "confidence_less": "[...] me seem less confident in my opinion [...]",
    "knowledge_more": "[...] me seem more knowledgeable about an issue [...]",
    "knowledge_less": "[...] me seem less knowledgeable about an issue [...]",
    "importance_more": "[...] an issue seem more important to me [...]",
    "importance_less": "[...] an issue seem less important to me [...]",
    "relevance_more": "[...] my writing seem more relevant or on-topic [...]",
    "relevance_less": "[...] my writing seem less relevant or on-topic [...]",
    "clarity_more": "[...] my opinion clearer to others [...]",
    "clarity_less": "[...] my opinion less clear others [...]",
    "formality_more": "[...] my writing seem more formal [...]",
    "formality_less": "[...] my writing seem less formal [...]",
    "informative_more": "[...] my writing seem more informative or educational [...]",
    "informative_less": "[...] my writing seem less informative or educational [...]",
    "originality_more": "[...] my writing seem more original or novel [...]",
    "originality_less": "[...] my writing seem less original or novel [...]",
    "friendliness_more": "[...] me seem more friendly [...]",
    "friendliness_less": "[...] me seem less friendly [...]",
    "optimism_more": "[...] me seem more optimistic [...]",
    "optimism_less": "[...] me seem less optimistic [...]",
    "community_more": "[...] me seem more community-oriented [...]",
    "community_less": "[...] me seem less community-oriented [...]",
    "open_views_more": "[...] me seem more open to reconsidering my views [...]",
    "open_views_less": "[...] me seem less open to reconsidering my views [...]",
    "hope_more": "[...] my writing express more hope [...]",
    "hope_less": "[...] my writing express less hope [...]",
    "excitement_more": "[...] my writing express more excitement [...]",
    "excitement_less": "[...] my writing express less excitement [...]",
    "fear_more": "[...] my writing express more fear [...]",
    "fear_less": "[...] my writing express less fear [...]",
    "disgust_more": "[...] my writing express more disgust [...]",
    "disgust_less": "[...] my writing express less disgust [...]",
    "anger_more": "[...] my writing express more anger [...]",
    "anger_less": "[...] my writing express less anger [...]",
    "age_older": "[...] me seem older [...]",
    "age_younger": "[...] me seem younger [...]",
    "gender_different": "[...] my gender seem different [...]",
    "race_different": "[...] my race or ethnicity seem different [...]",
    "english_better": "[...] my English language skills seem better [...]",
    "english_worse": "[...] my English language skills seem worse [...]",
    "education_more": "[...] me seem more educated [...]",
    "education_less": "[...] me seem less educated [...]",
    "income_higher": "[...] my income seem higher [...]",
    "income_lower": "[...] my income seem lower [...]",
    "politics_party_different": "[...] it seem like I voted for a different political party [...]",
    "politics_extreme": "[...] me seem more politically extreme [...]",
    "politics_moderate": "[...] me seem more politically moderate / centrist [...]",
}


def create_tolerance_whisker_plot(
    summary_df: pd.DataFrame,
    statistic: str = "mean",  # "mean" or "median"
    label_mapping: dict | None = None,
    figsize: tuple | None = None,
    save_path: str | None = None,
):

    value_col = statistic
    ci_low_col = f"{statistic}_ci_low"
    ci_high_col = f"{statistic}_ci_high"

    df = summary_df.copy()

    # Sort descending
    df = df.sort_values(value_col, ascending=False)

    # Labels
    if label_mapping is not None:
        labels = df["distortion"].map(label_mapping).fillna(df["distortion"])
    else:
        labels = df["distortion"]

    values = df[value_col].values
    lower_errors = values - df[ci_low_col].values
    upper_errors = df[ci_high_col].values - values

    # Plot
    if figsize is None:
        figsize = (10, max(5, 1.2 + 0.6 * len(df)))

    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for background bands
    colors = {
        "darkred": "#cc0000",  # 0-25
        "lightred": "#e06666",  # 25-50
        "lightgreen": "#93c47d",  # 50-75
        "darkgreen": "#38761d",  # 75-100
    }

    # Create background color bands
    y_min = -0.5
    y_max = len(df) - 0.5

    # Add colored rectangles for each range
    ax.add_patch(
        Rectangle(
            (0, y_min),
            25,
            y_max - y_min + 1,
            facecolor=colors["darkred"],
            alpha=0.3,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (25, y_min),
            25,
            y_max - y_min + 1,
            facecolor=colors["lightred"],
            alpha=0.3,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (50, y_min),
            25,
            y_max - y_min + 1,
            facecolor=colors["lightgreen"],
            alpha=0.3,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (75, y_min),
            25,
            y_max - y_min + 1,
            facecolor=colors["darkgreen"],
            alpha=0.3,
            zorder=0,
        )
    )

    y_positions = range(len(df))

    ax.errorbar(
        values,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=3,
        color="black",
        markerfacecolor="white",
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{statistic.capitalize()} Tolerance Score")
    ax.set_ylim(-0.5, len(df) - 0.5)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


for statistic in ["mean", "median"]:
    fig, ax = create_tolerance_whisker_plot(
        summary_df,
        statistic=statistic,  # or "median"
        label_mapping=distortion_label_mapping,
        save_path=os.path.join(
            FIGURES_DIR, f"distortion_tolerance_{statistic}.pdf"
        ),
    )
    plt.close(fig)

plt.close("all")
