"""Helpers for summarising LLM-coded free-text preference reasons (offline, no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_utils import get_group_offsets

PREFERENCE_LABELS = {"edited": "Preferred AI-edited", "original": "Preferred own original"}
PREFERENCE_COLORS = {"edited": "#1f77b4", "original": "#d62728"}
Z_95 = 1.959964
X_AXIS_PADDING = 5


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half_width, centre + half_width


def prevalence_table(df: pd.DataFrame, codes: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(group)
        for code in codes:
            k = int(group[code].sum())
            lower, upper = wilson_ci(k, n)
            rows.append({**dict(zip(group_cols, keys)), "code": code, "n": n, "n_mentions": k,
                         "pct": 100 * k / n, "ci_lower": 100 * lower, "ci_upper": 100 * upper})
    return pd.DataFrame(rows)


def plot_prevalence(prevalence: pd.DataFrame, codes: list[str], output_path: Path,
                    x_label: str = "% of responses mentioning reason") -> None:
    order = (prevalence.groupby("code")["pct"].mean().reindex(codes).sort_values().index.tolist())
    y_positions = {code: i for i, code in enumerate(order)}
    offsets = get_group_offsets(list(PREFERENCE_LABELS), offset_scale=0.25)

    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(order) + 1.2))
    for preference, label in PREFERENCE_LABELS.items():
        sub = prevalence.loc[prevalence["writer_preference"] == preference]
        y = sub["code"].map(y_positions) + offsets[preference]
        n = sub["n"].iloc[0]
        ax.errorbar(sub["pct"], y,
                    xerr=[sub["pct"] - sub["ci_lower"], sub["ci_upper"] - sub["pct"]],
                    fmt="o", color=PREFERENCE_COLORS[preference], capsize=2, markersize=5,
                    label=f"{label} (n={n})")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([code.replace("_", " ") for code in order])
    ax.set_xlabel(x_label)
    # Pad above the largest percentage, but never clip a confidence interval
    x_max = max(prevalence["pct"].max() + X_AXIS_PADDING, prevalence["ci_upper"].max() + 1)
    ax.set_xlim(0, min(100, x_max))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_preference_reasons(codes_path: Path, codebook_path: Path, results_dir: Path, figure_path: Path,
                                 by_condition: bool = True,
                                 x_label: str = "% of responses mentioning reason") -> None:
    """Write prevalence tables and figure for one study's `reason_codes.csv`."""
    with open(codebook_path) as f:
        codebook = json.load(f)
    codes = [entry["code"] for entry in codebook["codes"]]
    coded = pd.read_csv(codes_path)
    versions = coded["codebook_version"].astype(str).unique()
    if list(versions) != [str(codebook["version"])]:
        raise ValueError(f"reason_codes.csv has codebook versions {versions}, codebook is {codebook['version']}")

    print(f"Coded responses by preference:\n{coded['writer_preference'].value_counts().to_string()}")
    strict = coded.loc[coded["writer_preference"].isin(PREFERENCE_LABELS)]

    prevalence = prevalence_table(strict, codes, ["writer_preference"])
    tables = {"reason_prevalence.csv": prevalence}
    if by_condition:
        tables["reason_prevalence_by_condition.csv"] = prevalence_table(strict, codes, ["writer_preference", "condition"])
    rating_attributes = {entry["code"]: "; ".join(entry.get("rating_attributes", [])) for entry in codebook["codes"]}
    for table in tables.values():
        table.insert(table.columns.get_loc("code") + 1, "rating_attributes", table["code"].map(rating_attributes))
    n_codes = (coded.groupby(["writer_preference", "n_codes"]).size().rename("n").reset_index())
    n_codes["pct"] = 100 * n_codes["n"] / n_codes.groupby("writer_preference")["n"].transform("sum")

    results_dir.mkdir(parents=True, exist_ok=True)
    for file_name, table in tables.items():
        table.to_csv(results_dir / file_name, index=False)
    n_codes.to_csv(results_dir / "reason_n_codes.csv", index=False)
    plot_prevalence(prevalence, codes, figure_path, x_label=x_label)

    print("\nReason prevalence (%):")
    print(prevalence.pivot(index="code", columns="writer_preference", values="pct")
          .reindex(codes).round(1).to_string())
