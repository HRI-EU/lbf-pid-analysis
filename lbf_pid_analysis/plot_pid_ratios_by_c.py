#!/usr/bin/env python
#
# Plot selected derived SxPID quantities by heuristic and environment
# cooperativity, c.
#
# Trial-level inclusion
# ---------------------
# Only trials with significant joint mutual information I(T; S1, S2) are
# used when calculating PID summaries.
#
# Condition-level inclusion
# -------------------------
# A policy-c condition is included only if at least k trials have significant
# joint mutual information. The threshold k is controlled through:
#
#     --min-significant-trials
#
# For 50 independent trials tested at alpha = 0.05, k = 6 is the smallest
# count for which a one-sided binomial test gives
#
#     P(K >= k | K ~ Binomial(50, 0.05)) < 0.05.
#
# The module plots:
#
#   A-B: Observable jointness by c
#
#       J_obs = syn / (syn + shd)
#
#   C-D: Unique-to-collective balance by c
#
#       B_CU = (C - U) / (C + U)
#
#       C = syn + shd
#       U = unq1 + unq2
#
#   E-F: Collective-organization map
#
#       x = syn / (syn + shd)
#       y = (syn + shd) / (unq1 + unq2 + syn + shd)
#
#   G-H: Shared-information versus synergistic-information map
#
#       x = shd
#       y = syn
#
# Marker size is constant. Joint mutual information is used only to select
# statistically supported trials and conditions.
#
# Copyright (c) Honda Research Institute Europe GmbH
# This file is part of lbf_pid_analysis.
# lbf_pid_analysis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# lbf_pid_analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with lbf_pid_analysis. If not, see <http://www.gnu.org/licenses/>.
#
#
import argparse
import logging
import pickle
from pathlib import Path

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import initialize_logger

FILETYPE = "svg"  # "pdf"
DENOMINATOR_EPS = 1e-12

NON_COOPERATIVE_POLICIES = ["BL", "Ego"]
COOPERATIVE_POLICIES = ["Coop", "Adapt", "Social"]

ANNOTATED_C_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)

MARKER_SIZE = 32

BACKGROUND_LABEL_COLOR = "0.82"
BACKGROUND_LABEL_ALPHA = 1.0
BACKGROUND_LABEL_FONTSIZE = cfg.plot_elements["min_font_size"]


def get_plot_label(n_labels, plot_label_type="A"):
    """Return consecutive subplot labels."""
    return np.array([chr(ord(plot_label_type) + index) for index in range(n_labels)])


def get_policy_colors():
    """Return colors for all plotted policies."""
    return {
        "BL": cfg.colors["blue_muted"],
        "Ego": cfg.colors["blue_muted_lighter_1"],
        "Coop": cfg.colors["red_muted"],
        "Adapt": cfg.colors["green_muted"],
        "Social": cfg.colors["grey_muted"],
    }


def load_experiment_results(
    loadpath,
    target_variable,
    folder_steps=5,
    max_folder=25,
):
    """Load experiment results into a dictionary indexed by c."""
    results = {}
    coop_parameters = []
    folders = np.array([1, folder_steps])

    logging.info("Using target variable %s", target_variable)
    logging.info("Loading data from %s", loadpath)

    while folders[1] < (max_folder + 1):
        filename = (
            f"mi_estimates_over_trials_"
            f"{folders[0]}_to_{folders[-1]}_"
            f"t_{target_variable}.p"
        )

        logging.info(
            "Reading results for folders %s to %s: %s",
            folders[0],
            folders[1],
            filename,
        )

        with open(loadpath.joinpath(filename), "rb") as file:
            data = pickle.load(file)

        coop = float(data["experiment_settings"]["environment"]["coop"])

        coop_parameters.append(coop)
        results[coop] = data

        folders[0] = folders[1] + 1
        folders[1] = folders[1] + folder_steps

    return results, sorted(coop_parameters)


def safe_ratio(
    numerator,
    denominator,
    eps=DENOMINATOR_EPS,
):
    """Calculate a ratio and return NaN for near-zero denominators."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    numerator, denominator = np.broadcast_arrays(
        numerator,
        denominator,
    )

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > eps)
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=valid,
    )

    return result


def get_measure_values(
    results,
    coop,
    measure,
    heuristic,
    dtype=float,
):
    """Return flattened trial values for one stored measure."""
    if measure not in results[coop]:
        raise KeyError(
            f"Measure '{measure}' is not available for c={coop}. "
            f"Available keys are: {list(results[coop].keys())}"
        )

    if heuristic not in results[coop][measure]:
        raise KeyError(
            f"Heuristic '{heuristic}' is not available for measure "
            f"'{measure}' and c={coop}."
        )

    return np.asarray(
        results[coop][measure][heuristic],
        dtype=dtype,
    ).ravel()


def convert_significance_values(values):
    """Convert stored significance indicators to Boolean values."""
    values = np.asarray(values).ravel()

    if np.issubdtype(values.dtype, np.bool_):
        return values.astype(bool)

    if np.issubdtype(values.dtype, np.number):
        return values.astype(float) != 0

    true_values = {
        "true",
        "t",
        "1",
        "yes",
        "y",
        "significant",
        "sig",
    }

    false_values = {
        "false",
        "f",
        "0",
        "no",
        "n",
        "not significant",
        "nonsignificant",
        "non-significant",
        "ns",
    }

    converted = []

    for value in values:
        normalized = str(value).strip().lower()

        if normalized in true_values:
            converted.append(True)
        elif normalized in false_values:
            converted.append(False)
        else:
            raise ValueError(
                f"Could not interpret joint-MI significance value '{value}'."
            )

    return np.asarray(converted, dtype=bool)


def create_derived_dataframe(
    results: dict,
    coop_parameters: list,
    estimator: str,
    alpha: float = 0.05,
    min_significant_trials: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create derived PID quantities from supported conditions.

    First, trials are retained if ``jmi_sign`` is true.

    Second, a complete policy-c condition is retained only if the number of
    significant trials is at least ``min_significant_trials``.

    Parameters
    ----------
    results : dict
        Loaded result dictionary.
    coop_parameters : sequence of float
        Values of environment cooperativity c.
    estimator : str
        PID estimator used for estimation
    alpha : float
        Expected trial-level significance threshold. Filtering uses the stored
        significance decision; alpha is used for consistency checks.
    min_significant_trials : int
        Minimum number of significant trials required to include a complete
        policy-c condition.

    Returns
    -------
    plot_data : pandas.DataFrame
        One row per retained trial.
    filter_summary : pandas.DataFrame
        Trial counts and condition-level inclusion decisions.
    """
    if min_significant_trials < 1:
        raise ValueError("min_significant_trials must be at least 1.")

    syn_measure = f"syn_norm_{estimator}"
    shd_measure = f"shd_norm_{estimator}"
    unq1_measure = f"unq1_norm_{estimator}"
    unq2_measure = f"unq2_norm_{estimator}"
    jmi_measure = "jmi"
    jmi_sign_measure = "jmi_sign"
    jmi_p_measure = "jmi_p"

    rows = []
    filter_rows = []

    required_measures = [
        syn_measure,
        shd_measure,
        unq1_measure,
        unq2_measure,
        jmi_measure,
        jmi_sign_measure,
        jmi_p_measure,
    ]

    n_invalid_collective = 0
    n_invalid_total = 0
    n_inconsistent_significance = 0

    for coop in coop_parameters:
        for measure in required_measures:
            if measure not in results[coop]:
                raise KeyError(
                    f"Measure '{measure}' is not available for c={coop}. "
                    f"Available keys are: {list(results[coop].keys())}"
                )

        heuristics = list(results[coop][syn_measure].keys())

        for measure in required_measures[1:]:
            if set(results[coop][measure].keys()) != set(heuristics):
                raise ValueError(
                    f"Policies differ between '{syn_measure}' and "
                    f"'{measure}' for c={coop}."
                )

        for heuristic in heuristics:
            syn_values = get_measure_values(
                results,
                coop,
                syn_measure,
                heuristic,
            )

            shd_values = get_measure_values(
                results,
                coop,
                shd_measure,
                heuristic,
            )

            unq1_values = get_measure_values(
                results,
                coop,
                unq1_measure,
                heuristic,
            )

            unq2_values = get_measure_values(
                results,
                coop,
                unq2_measure,
                heuristic,
            )

            jmi_values = get_measure_values(
                results,
                coop,
                jmi_measure,
                heuristic,
            )

            jmi_sign_raw = get_measure_values(
                results,
                coop,
                jmi_sign_measure,
                heuristic,
                dtype=None,
            )

            jmi_sign_values = convert_significance_values(jmi_sign_raw)

            jmi_p_values = get_measure_values(
                results,
                coop,
                jmi_p_measure,
                heuristic,
            )

            arrays = [
                syn_values,
                shd_values,
                unq1_values,
                unq2_values,
                jmi_values,
                jmi_sign_values,
                jmi_p_values,
            ]

            shapes = {array.shape for array in arrays}

            if len(shapes) != 1:
                raise ValueError(
                    "Different numbers of estimates for heuristic "
                    f"'{heuristic}', c={coop}: "
                    f"{[array.shape for array in arrays]}"
                )

            p_value_decision = np.isfinite(jmi_p_values) & (jmi_p_values < alpha)

            inconsistent = np.isfinite(jmi_p_values) & (
                jmi_sign_values != p_value_decision
            )

            n_inconsistent_significance += np.count_nonzero(inconsistent)

            significant = jmi_sign_values & np.isfinite(jmi_values)

            n_total = significant.size
            n_significant = int(np.count_nonzero(significant))

            fraction_significant = n_significant / n_total if n_total > 0 else np.nan

            condition_included = n_significant >= min_significant_trials

            filter_rows.append(
                {
                    "c": coop,
                    "heuristic": heuristic,
                    "policy": cfg.labels[heuristic],
                    "n_total": n_total,
                    "n_significant": n_significant,
                    "n_excluded_trials": (n_total - n_significant),
                    "fraction_significant": fraction_significant,
                    "minimum_required": min_significant_trials,
                    "condition_included": condition_included,
                }
            )

            if condition_included:
                logging.info(
                    "%s, c=%.1f: including condition with %d/%d "
                    "significant joint-MI trials",
                    cfg.labels[heuristic],
                    coop,
                    n_significant,
                    n_total,
                )
            else:
                logging.warning(
                    "%s, c=%.1f: excluding condition; only %d/%d "
                    "trials have significant joint MI, but at least %d "
                    "are required",
                    cfg.labels[heuristic],
                    coop,
                    n_significant,
                    n_total,
                    min_significant_trials,
                )
                continue

            retained_trial_indices = np.flatnonzero(significant)

            syn_values = syn_values[significant]
            shd_values = shd_values[significant]
            unq1_values = unq1_values[significant]
            unq2_values = unq2_values[significant]
            jmi_values = jmi_values[significant]
            jmi_p_values = jmi_p_values[significant]

            individual = unq1_values + unq2_values

            collective = syn_values + shd_values

            total_pid = individual + collective

            observable_jointness = safe_ratio(
                syn_values,
                collective,
            )

            collective_fraction = safe_ratio(
                collective,
                total_pid,
            )

            unique_collective_balance = safe_ratio(
                collective - individual,
                total_pid,
            )

            n_invalid_collective += np.count_nonzero(~np.isfinite(observable_jointness))

            n_invalid_total += np.count_nonzero(~np.isfinite(collective_fraction))

            for retained_index, original_trial_index in enumerate(
                retained_trial_indices
            ):
                rows.append(
                    {
                        "c": coop,
                        "trial": int(original_trial_index),
                        "heuristic": heuristic,
                        "policy": cfg.labels[heuristic],
                        "syn": syn_values[retained_index],
                        "shd": shd_values[retained_index],
                        "unq1": unq1_values[retained_index],
                        "unq2": unq2_values[retained_index],
                        "individual": individual[retained_index],
                        "collective": collective[retained_index],
                        "total_pid": total_pid[retained_index],
                        "jmi": jmi_values[retained_index],
                        "jmi_p": jmi_p_values[retained_index],
                        "jmi_sign": True,
                        "observable_jointness": (observable_jointness[retained_index]),
                        "collective_fraction": (collective_fraction[retained_index]),
                        "unique_collective_balance": (
                            unique_collective_balance[retained_index]
                        ),
                    }
                )

    if n_inconsistent_significance:
        logging.warning(
            "%d stored jmi_sign decisions were inconsistent with "
            "jmi_p < %.4f. Filtering used the stored jmi_sign values.",
            n_inconsistent_significance,
            alpha,
        )

    if n_invalid_collective:
        logging.warning(
            "%d retained trials had undefined observable jointness "
            "because |syn + shd| <= %.2e.",
            n_invalid_collective,
            DENOMINATOR_EPS,
        )

    if n_invalid_total:
        logging.warning(
            "%d retained trials had undefined collective fractions "
            "because |unq1 + unq2 + syn + shd| <= %.2e.",
            n_invalid_total,
            DENOMINATOR_EPS,
        )

    plot_data = pd.DataFrame(rows)
    filter_summary = pd.DataFrame(filter_rows)

    if plot_data.empty:
        raise ValueError(
            "No policy-c conditions passed the minimum significant-trial "
            f"threshold of {min_significant_trials}."
        )

    return plot_data, filter_summary


def summarize_for_plotting(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate condition means and standard deviations."""
    return (
        data.groupby(
            ["policy", "c"],
            as_index=False,
        )
        .agg(
            n_trials=("trial", "size"),
            syn=("syn", "mean"),
            syn_sd=("syn", "std"),
            shd=("shd", "mean"),
            shd_sd=("shd", "std"),
            jmi=("jmi", "mean"),
            observable_jointness=(
                "observable_jointness",
                "mean",
            ),
            observable_jointness_sd=(
                "observable_jointness",
                "std",
            ),
            collective_fraction=(
                "collective_fraction",
                "mean",
            ),
            collective_fraction_sd=(
                "collective_fraction",
                "std",
            ),
            unique_collective_balance=(
                "unique_collective_balance",
                "mean",
            ),
            unique_collective_balance_sd=(
                "unique_collective_balance",
                "std",
            ),
        )
        .sort_values(["policy", "c"])
    )


def set_plotlabel(
    axis,
    label,
    x_pos=-0.15,
    y_pos=1.01,
):
    """Add a bold subplot label."""
    axis.text(
        x_pos,
        y_pos,
        label,
        transform=axis.transAxes,
        fontsize=7,
        fontweight="bold",
    )


def should_annotate_c(value):
    """Return whether c should be annotated."""
    return any(np.isclose(value, selected) for selected in ANNOTATED_C_VALUES)


def annotate_trajectory(
    axis,
    policy_data,
    x_column,
    y_column,
    color,
):
    """Annotate selected values of c along a trajectory."""
    for _, row in policy_data.iterrows():
        if not should_annotate_c(row["c"]):
            continue

        if not (np.isfinite(row[x_column]) and np.isfinite(row[y_column])):
            continue

        axis.annotate(
            f"{row['c']:.1f}",
            (
                row[x_column],
                row[y_column],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5,
            color=color,
        )


def add_policy_legend(
    axis,
    loc="best",
):
    """Add a policy legend without duplicate entries."""
    handles, labels = axis.get_legend_handles_labels()

    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    axis.legend(
        unique.values(),
        unique.keys(),
        loc=loc,
        title=None,
    )


def plot_index_by_c(
    summary: pd.DataFrame,
    index: str,
    index_sd: str,
    ylabel: str,
    policies: list[str],
    axis,
    subplot_label,
    reference=0,
    ylim=None,
    show_policy_legend=False,
):
    """Plot one derived index as a function of c."""
    colors = get_policy_colors()

    for policy in policies:
        policy_data = summary[summary["policy"] == policy].dropna(subset=[index]).copy()

        if policy_data.empty:
            continue

        x = policy_data["c"].to_numpy()
        y = policy_data[index].to_numpy()
        y_sd = policy_data[index_sd].fillna(0).to_numpy()

        color = colors[policy]

        axis.fill_between(x, y - y_sd, y + y_sd, color=color, alpha=0.18, linewidth=0)
        axis.plot(x, y, color=color, linewidth=1.5, label=policy)
        axis.scatter(
            x, y, s=MARKER_SIZE, color=color, edgecolor="white", linewidth=0.5, zorder=3
        )

    if reference is not None:
        axis.axhline(reference, zorder=0, color="lightgray", linestyle=":", linewidth=1)

    axis.set(xlabel="$c$", ylabel=ylabel)

    if ylim is not None:
        axis.set_ylim(ylim)
    axis.margins(x=0.03, y=0.06)

    set_plotlabel(axis, subplot_label)

    if show_policy_legend:
        add_policy_legend(axis, loc="best")


def add_collective_map_background(axis):
    """Add faint labels to the collective-organization map."""
    axis.axvline(0.5, zorder=0, color="0.88", linestyle=":", linewidth=1)
    axis.axhline(0.5, zorder=0, color="0.88", linestyle=":", linewidth=1)


def add_synergy_shared_background(axis):
    """Add faint labels to the shared-versus-synergy map."""
    axis.plot(
        [0, 1],
        [1, 0],
        transform=axis.transAxes,
        zorder=0,
        color="0.90",
        linestyle=":",
        linewidth=1,
    )


def plot_trajectory_map(
    summary,
    policies,
    axis,
    subplot_label,
    x_column,
    y_column,
    xlabel,
    ylabel,
    add_background,
    show_policy_legend=False,
    policy_legend_loc="best",
):
    """Plot a two-dimensional policy trajectory."""
    colors = get_policy_colors()

    for policy in policies:
        policy_data = (
            summary[summary["policy"] == policy]
            .dropna(subset=[x_column, y_column])
            .copy()
        )

        if policy_data.empty:
            continue

        color = colors[policy]

        axis.plot(
            policy_data[x_column],
            policy_data[y_column],
            color=color,
            linewidth=1.5,
            label=policy,
        )

        axis.scatter(
            policy_data[x_column],
            policy_data[y_column],
            s=MARKER_SIZE,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        annotate_trajectory(
            axis=axis,
            policy_data=policy_data,
            x_column=x_column,
            y_column=y_column,
            color=color,
        )

    add_background(axis)
    margin = 0.06
    axis.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(-margin, 1 + margin),
        ylim=(-margin, 1 + margin),
    )
    axis.set_aspect("equal", adjustable="box")
    set_plotlabel(axis, subplot_label)
    if show_policy_legend:
        add_policy_legend(axis, loc=policy_legend_loc)


def main(
    path: str,
    target_variable: str,
    estimator: str,
    alpha: float,
    min_significant_trials: int,
    n_folders: int,
    render: bool,
) -> None:
    """Load supported conditions, calculate indices, and plot results."""
    initialize_logger(
        log_name=(f"compare_selected_pid_indices_significant_jmi_t_{target_variable}")
    )

    sns.set_style("ticks")

    loadpath = Path(path).joinpath("results_single_trial_analysis")

    results, coop_parameters = load_experiment_results(
        loadpath=loadpath,
        target_variable=target_variable,
        max_folder=n_folders,
    )

    plot_data, filter_summary = create_derived_dataframe(
        results=results,
        coop_parameters=coop_parameters,
        estimator=estimator,
        alpha=alpha,
        min_significant_trials=min_significant_trials,
    )

    filter_filename = loadpath.joinpath(
        f"significant_jmi_condition_support_t_{target_variable}.csv"
    )

    filter_summary.to_csv(filter_filename, index=False)
    logging.info("Saved condition-support summary to %s", filter_filename)

    summary = summarize_for_plotting(plot_data)
    figure = plt.figure(
        figsize=(
            cfg.plot_elements["max_width_in"],
            2 * cfg.plot_elements["row_height"],
        ),
    )
    gs = figure.add_gridspec(nrows=2, ncols=1)
    # Top row: add margins around A and B
    # Bottom row: four equally sized plots
    gs_top = gs[0].subgridspec(
        nrows=1,
        ncols=4,
        width_ratios=[0.1, 1, 1, 0.1],
    )
    gs_bottom = gs[1].subgridspec(
        nrows=1,
        ncols=4,
    )
    axes = [
        figure.add_subplot(gs_top[0, 1]),
        figure.add_subplot(gs_top[0, 2]),
        figure.add_subplot(gs_bottom[0, 0]),
        figure.add_subplot(gs_bottom[0, 1]),
        figure.add_subplot(gs_bottom[0, 2]),
        figure.add_subplot(gs_bottom[0, 3]),
    ]

    subplot_labels = get_plot_label(6)

    # A-B: Unique-to-collective balance
    plot_index_by_c(
        summary=summary,
        index="unique_collective_balance",
        index_sd="unique_collective_balance_sd",
        ylabel=(r"$(C-U)/(C+U)$"),
        policies=NON_COOPERATIVE_POLICIES,
        axis=axes[0],
        subplot_label=subplot_labels[0],
        reference=0,
        ylim=(-1.08, 1.08),
        show_policy_legend=False,
    )

    plot_index_by_c(
        summary=summary,
        index="unique_collective_balance",
        index_sd="unique_collective_balance_sd",
        ylabel=(r"$(C-U)/(C+U)$"),
        policies=COOPERATIVE_POLICIES,
        axis=axes[1],
        subplot_label=subplot_labels[1],
        reference=0,
        ylim=(-1.08, 1.08),
        show_policy_legend=False,
    )

    # # C-D: Observable jointness
    # plot_index_by_c(
    #     summary=summary,
    #     index="observable_jointness",
    #     index_sd="observable_jointness_sd",
    #     ylabel=("$I_{\mathrm{syn}}/(I_{\mathrm{syn}}+I_{\mathrm{shd}})$"),
    #     policies=NON_COOPERATIVE_POLICIES,
    #     axis=axes[1, 0],
    #     subplot_label=subplot_labels[0, 0],
    #     reference=0,
    #     ylim=(-0.05, 1.08),
    #     show_policy_legend=False,
    # )

    # plot_index_by_c(
    #     summary=summary,
    #     index="observable_jointness",
    #     index_sd="observable_jointness_sd",
    #     ylabel=("$I_{\mathrm{syn}}/(I_{\mathrm{syn}}+I_{\mathrm{shd}})$"),
    #     policies=COOPERATIVE_POLICIES,
    #     axis=axes[1, 1],
    #     subplot_label=subplot_labels[0, 1],
    #     reference=0,
    #     ylim=(-0.05, 1.08),
    #     show_policy_legend=True,
    # )

    # C-D: Collective-organization maps
    plot_trajectory_map(
        summary=summary,
        policies=NON_COOPERATIVE_POLICIES,
        axis=axes[2],
        subplot_label=subplot_labels[2],
        x_column="observable_jointness",
        y_column="collective_fraction",
        xlabel=(cfg.labels_short[f"syn_{estimator}"] + r"$/C$"),
        ylabel=(r"$C/I(G;A_1,A_2)$"),
        add_background=add_collective_map_background,
        show_policy_legend=True,
        policy_legend_loc="upper left",
    )

    plot_trajectory_map(
        summary=summary,
        policies=COOPERATIVE_POLICIES,
        axis=axes[3],
        subplot_label=subplot_labels[3],
        x_column="observable_jointness",
        y_column="collective_fraction",
        xlabel=(cfg.labels_short[f"syn_{estimator}"] + r"$/C$"),
        ylabel=(r"$C/I(G;A_1,A_2)$"),
        add_background=add_collective_map_background,
        show_policy_legend=False,
    )

    # G-H: Shared versus synergy
    plot_trajectory_map(
        summary=summary,
        policies=NON_COOPERATIVE_POLICIES,
        axis=axes[4],
        subplot_label=subplot_labels[4],
        x_column="shd",
        y_column="syn",
        xlabel=(cfg.labels_short[f"shd_{estimator}"] + r"$/I(G;A_1,A_2)$"),
        ylabel=(cfg.labels_short[f"syn_{estimator}"] + r"$/I(G;A_1,A_2)$"),
        # xlabel=(r"$I_{\mathrm{shd}}/I(G;A_1,A_2)$"),
        # ylabel=(r"$I_{\mathrm{syn}}/I(G;A_1,A_2)$"),
        add_background=add_synergy_shared_background,
        show_policy_legend=True,
    )

    plot_trajectory_map(
        summary=summary,
        policies=COOPERATIVE_POLICIES,
        axis=axes[5],
        subplot_label=subplot_labels[5],
        x_column="shd",
        y_column="syn",
        xlabel=(cfg.labels_short[f"shd_{estimator}"] + r"$/I(G;A_1,A_2)$"),
        ylabel=(cfg.labels_short[f"syn_{estimator}"] + r"$/I(G;A_1,A_2)$"),
        # xlabel=(r"$I_{\mathrm{shd}}/I(G;A_1,A_2)$"),
        # ylabel=(r"$I_{\mathrm{syn}}/I(G;A_1,A_2)$"),
        add_background=add_synergy_shared_background,
        show_policy_legend=True,
    )

    for axis in axes[2:]:
        axis.set_aspect("equal", adjustable="box")

    figure.tight_layout()
    figure.subplots_adjust(hspace=0.58, wspace=0.42)

    filename = loadpath.joinpath(
        f"pid_ratios_t_{estimator}_{target_variable}.{FILETYPE}"
    )

    logging.info("Saving figure to %s", filename)
    figure.savefig(filename)

    if render:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected derived PID quantities using only conditions "
            "with a minimum number of significant joint-MI trials."
        )
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the figure after saving it.",
    )

    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiment_results",
        type=str,
        help="Path to the experiment directory.",
    )

    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help="Target variable used for PID estimation.",
    )

    parser.add_argument(
        "--syn-measure",
        default="syn_norm_sx_cor",
        type=str,
        help=("Result key containing relative synergistic information."),
    )

    parser.add_argument(
        "--shd-measure",
        default="shd_norm_sx_cor",
        type=str,
        help=("Result key containing relative shared information."),
    )

    parser.add_argument(
        "--unq1-measure",
        default="unq1_norm_sx_cor",
        type=str,
        help=("Result key containing relative unique information for source 1."),
    )

    parser.add_argument(
        "--unq2-measure",
        default="unq2_norm_sx_cor",
        type=str,
        help=("Result key containing relative unique information for source 2."),
    )

    parser.add_argument(
        "--jmi-measure",
        default="jmi",
        type=str,
        help="Result key containing absolute I(T; S1, S2).",
    )

    parser.add_argument(
        "--jmi-sign-measure",
        default="jmi_sign",
        type=str,
        help=(
            "Result key containing the stored significance decision for I(T; S1, S2)."
        ),
    )

    parser.add_argument(
        "--jmi-p-measure",
        default="jmi_p",
        type=str,
        help=("Result key containing the p-value of the joint-MI test."),
    )

    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help=(
            "Expected trial-level significance threshold. Filtering uses "
            "jmi_sign; alpha checks consistency with jmi_p."
        ),
    )

    parser.add_argument(
        "--min-significant-trials",
        default=6,
        type=int,
        help=(
            "Minimum number of significant joint-MI trials required to "
            "include a complete policy-c condition. For 50 trials tested "
            "at alpha=0.05, k=6 is the smallest count exceeding the "
            "one-sided binomial null at the 0.05 level."
        ),
    )

    parser.add_argument(
        "-f",
        "--folders",
        default=25,
        type=int,
        help="Number of experiment result folders to parse.",
    )

    arguments = parser.parse_args()

    main(
        path=arguments.path,
        target_variable=arguments.target,
        syn_measure=arguments.syn_measure,
        shd_measure=arguments.shd_measure,
        unq1_measure=arguments.unq1_measure,
        unq2_measure=arguments.unq2_measure,
        jmi_measure=arguments.jmi_measure,
        jmi_sign_measure=arguments.jmi_sign_measure,
        jmi_p_measure=arguments.jmi_p_measure,
        alpha=arguments.alpha,
        min_significant_trials=arguments.min_significant_trials,
        n_folders=arguments.folders,
        render=arguments.render,
    )
