#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Plot local SxPID for selected trials.
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
import glob
import logging
from pathlib import Path

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_pid_per_trial import (
    generate_output_path,
    get_heuristic_used,
)
from matplotlib.gridspec import GridSpec
from plot_measure_by_c import load_experiment_results
from utils import initialize_logger, read_settings

FIGTYPE = "svg"  # "pdf"/"png"

# Keep one fixed realization order in every panel. Missing realizations are
# shown as empty rows, which makes conditions and trials directly comparable.
REALIZATION_ORDER = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
]

PROBABILITY_COLUMN = "p(s1,s2,t)"
WEIGHTED_COLUMNS = ["unq_s1*p", "unq_s2*p", "shd*p", "syn*p"]
GLOBAL_LABELS = [
    r"$I_{\mathrm{unq},1}$",
    r"$I_{\mathrm{unq},2}$",
    r"$I_{\mathrm{shd}}$",
    r"$I_{\mathrm{syn}}$",
]
LOCAL_LABELS = [
    r"$p\,i_{\mathrm{unq},1}$",
    r"$p\,i_{\mathrm{unq},2}$",
    r"$p\,i_{\mathrm{shd}}$",
    r"$p\,i_{\mathrm{syn}}$",
]
ATOM_COLORS = [
    cfg.colors["green_muted_lighter_1"],
    cfg.colors["green_muted_lighter_1"],
    cfg.colors["red_muted_lighter_1"],
    cfg.colors["blue_muted_lighter_1"],
]


def main(path, folder_number, target_variable, n_folders, render=False):
    initialize_logger(
        log_name="plot_local_sx_pid_t_{}_folder_{}".format(
            target_variable, folder_number
        ),
    )
    outpath = generate_output_path(path)
    local_pid_outpath = Path(outpath).joinpath("local_sx_pid_plots")
    local_pid_outpath.mkdir(parents=True, exist_ok=True)

    # Identify files collected for experiment. Load settings file.
    foldername = Path(path, f"{int(folder_number):02d}*")
    experiment_settings = read_settings(
        glob.glob(str(foldername.joinpath("experiment_settings.yml")))[0]
    )
    heuristic = get_heuristic_used(experiment_settings)
    coop_param = float(experiment_settings["environment"]["coop"])

    # Plot means and standard deviations across trials.
    results, _ = load_experiment_results(
        outpath, target_variable, max_folder=n_folders, load_local_pid=True
    )
    plot_local_pid_stats(
        results[coop_param]["local_sx_pid"][heuristic],
        local_pid_outpath,
        filename=f"local_sx_pid_{heuristic}_c_{coop_param:.2f}_t_{target_variable}",
        figuretitle=f"{cfg.labels[heuristic]}, $c={coop_param:.2f}$",
        render=render,
    )


def _prepare_local_pid_df(df):
    """Return one row per realization in a fixed, comparable order."""
    dat = df.copy()
    dat["realization"] = list(
        zip(
            dat[cfg.col_s1].astype(int),
            dat[cfg.col_s2].astype(int),
            dat[cfg.col_t].astype(int),
        )
    )
    dat = dat.set_index("realization")
    dat = dat.reindex(REALIZATION_ORDER)

    # Missing realizations have probability zero and therefore zero weighted
    # contribution to every global atom.
    fill_zero = [PROBABILITY_COLUMN, "mi*p"] + WEIGHTED_COLUMNS
    for column in fill_zero:
        if column in dat.columns:
            dat[column] = dat[column].fillna(0.0)

    return dat


def _symmetric_limit(values, minimum=0.01, pad=1.15):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return minimum
    return max(minimum, np.max(np.abs(values)) * pad)


def _plot_global_profile(ax, global_values, global_errors=None):
    """Plot the global PID profile above the realization-level panels."""
    x = np.arange(len(global_values))
    ax.bar(x, global_values, color=ATOM_COLORS)
    if global_errors is not None:
        ax.errorbar(
            x,
            global_values,
            yerr=global_errors,
            fmt="none",
            capsize=2,
            elinewidth=0.6,
            color="k",
        )
    ax.axhline(0, lw=0.5, color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(GLOBAL_LABELS)
    ax.set_ylabel(cfg.labels["sx"])


def _plot_realization_rows(
    fig,
    gs,
    probabilities,
    weighted_means,
    weighted_errors=None,
    probability_errors=None,
):
    """Plot probability and weighted local PID rows with shared scales."""
    n_rows = len(REALIZATION_ORDER)
    prob_max = max(
        0.05,
        np.nanmax(
            probabilities
            + (probability_errors if probability_errors is not None else 0)
        )
        * 1.15,
    )

    all_weighted = weighted_means.to_numpy().ravel()
    if weighted_errors is not None:
        all_weighted = np.concatenate(
            [
                all_weighted,
                (weighted_means + weighted_errors).to_numpy().ravel(),
                (weighted_means - weighted_errors).to_numpy().ravel(),
            ]
        )
    weighted_lim = _symmetric_limit(all_weighted)

    prob_axes = []
    pid_axes = []
    for row, realization in enumerate(REALIZATION_ORDER):
        ax_prob = fig.add_subplot(gs[row, 0])
        ax_pid = fig.add_subplot(gs[row, 1])
        prob_axes.append(ax_prob)
        pid_axes.append(ax_pid)

        # Probability bar.
        p = probabilities.iloc[row]
        p_err = None if probability_errors is None else probability_errors.iloc[row]
        ax_prob.barh([0], [p], color="lightgray")
        ax_prob.set(ylim=[-0.8, 0.8])
        if p_err is not None:
            ax_prob.errorbar(
                p,
                0,
                xerr=p_err,
                fmt="none",
                capsize=2,
                elinewidth=0.6,
                color="k",
            )
        ax_prob.set_xlim(0, prob_max)
        ax_prob.set_yticks([0])
        ax_prob.set_yticklabels([str(realization)], fontsize=6)
        ax_prob.axvline(0, lw=0.7, color="k")

        # Weighted local PID bars.
        values = weighted_means.iloc[row].to_numpy(dtype=float)
        x = np.arange(len(WEIGHTED_COLUMNS))
        ax_pid.bar(x, values, color=ATOM_COLORS)  # , width=0.3)
        if weighted_errors is not None:
            ax_pid.errorbar(
                x,
                values,
                yerr=weighted_errors.iloc[row].to_numpy(dtype=float),
                fmt="none",
                capsize=2,
                elinewidth=0.6,
                color="k",
            )
        ax_pid.axhline(0, lw=0.5, color="lightgray")
        ax_pid.set_ylim(-weighted_lim, weighted_lim)
        ax_pid.set_xticks(x)

        if row < n_rows - 1:
            ax_prob.tick_params(axis="x", labelbottom=False)
            ax_pid.tick_params(axis="x", labelbottom=False)
        else:
            ax_prob.set_xlabel(r"$p(a_1,a_2,g)$")
            ax_pid.set_xticklabels(LOCAL_LABELS, rotation=35, ha="right")

        # if row == 0:
        #     ax_prob.set_title("Empirical probability", fontsize=7)
        #     ax_pid.set_title("Probability-weighted local PID", fontsize=7)
        # else:
        # ax_pid.set_xticklabels([])

        ax_pid.tick_params(axis="y", labelsize=6)
        ax_prob.tick_params(axis="x", labelsize=6)

    return prob_axes, pid_axes


def plot_local_pid_stats(
    local_pid,
    outpath,
    filename,
    figuretitle=None,
    weighted=True,
    render=False,
):
    """Plot average local PID statistics across trials.

    Parameters
    ----------
    local_pid : list[pandas.DataFrame]
        Local SxPID results, one data frame per trial.
    outpath : pathlib.Path
        Output directory.
    filename : str
        Output filename without extension.
    figuretitle : str, optional
        Figure title.
    weighted : bool, optional
        Kept for API compatibility. The revised figure always uses weighted
        local contributions because these sum to the global PID atoms.
    render : bool, optional
        Whether to display the figure before closing it.
    """
    if not weighted:
        logging.warning(
            "The revised local PID figure always plots probability-weighted "
            "local contributions. Ignoring weighted=False."
        )

    prepared = [_prepare_local_pid_df(df) for df in local_pid]
    stacked = pd.concat(
        prepared, keys=np.arange(len(prepared)), names=["trial", "realization"]
    )

    df_means = stacked.groupby(level="realization").mean(numeric_only=True)
    df_sds = (
        stacked.groupby(level="realization").std(ddof=1, numeric_only=True).fillna(0)
    )
    df_counts = stacked.groupby(level="realization").count()

    df_means = df_means.reindex(REALIZATION_ORDER)
    df_sds = df_sds.reindex(REALIZATION_ORDER).fillna(0)
    df_counts = df_counts.reindex(REALIZATION_ORDER).fillna(0)

    # logging.info(
    #     "Statistics for local PID estimates over all trials:\nMeans:\n%s\nSD:\n%s\nCounts\n%s",
    #     df_means,
    #     df_sds,
    #     df_counts,
    # )
    logging.info("Saving statistics to %s/%s*.csv", outpath, filename)
    df_means.to_csv(outpath.joinpath(f"{filename}_means.csv"))
    df_sds.to_csv(outpath.joinpath(f"{filename}_sds.csv"))
    df_counts.to_csv(outpath.joinpath(f"{filename}_counts.csv"))

    probabilities = df_means[PROBABILITY_COLUMN]
    probability_errors = df_sds[PROBABILITY_COLUMN]
    weighted_means = df_means[WEIGHTED_COLUMNS]
    weighted_errors = df_sds[WEIGHTED_COLUMNS]

    # The global atoms are sums over weighted local contributions. Compute the
    # global profile per trial first, then take its mean and SD across trials.
    global_by_trial = pd.DataFrame(
        [df[WEIGHTED_COLUMNS].sum(axis=0) for df in prepared]
    )
    global_means = global_by_trial.mean(axis=0).to_numpy()
    global_sds = global_by_trial.std(axis=0, ddof=1).fillna(0).to_numpy()

    fig = plt.figure(figsize=(2, 3.5))
    gs_outer = GridSpec(2, 1, height_ratios=[1.2, 6.0], hspace=0.35, figure=fig)
    ax_global = fig.add_subplot(gs_outer[0])
    _plot_global_profile(ax_global, global_means, global_sds)

    gs_rows = gs_outer[1].subgridspec(
        len(REALIZATION_ORDER),
        2,
        width_ratios=[2, 4.0],
        wspace=0.4,
        hspace=0.0,
    )
    _plot_realization_rows(
        fig,
        gs_rows,
        probabilities,
        weighted_means,
        weighted_errors=weighted_errors,
        probability_errors=probability_errors,
    )

    if figuretitle:
        fig.suptitle(figuretitle, y=0.995, fontsize=8, fontweight="bold")

    plt.subplots_adjust(left=0.14, right=0.99, top=0.94, bottom=0.10)
    output_file = outpath.joinpath(f"{filename}_all_trials_stats.{FIGTYPE}")
    logging.info("Saving figure to %s", output_file)
    plt.savefig(output_file)
    if render:
        plt.show()
    else:
        plt.close()
