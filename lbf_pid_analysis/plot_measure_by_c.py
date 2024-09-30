#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Plot estimated information-theoretic or game performance measure by heuristic and
# cooperativity, c.
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
import logging
import pickle
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import initialize_logger
import config as cfg

FILETYPE = "pdf" # "png"


def get_plot_label(n_labels, plot_label_type="A"):
    return np.array([chr(ord(plot_label_type) + a) for a in range(n_labels)])


def set_layout():
    """Set matplotlib layout"""
    SMALL = 10
    MEDIUM = 12
    LARGE = 14
    plt.rc("font", size=MEDIUM)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM)  # legend fontsize
    plt.rc("figure", titlesize=LARGE)  # fontsize of the figure title


def get_colors(measure):
    COLORS = {
        "syn": [
            cfg.colors["blue_muted"],
            cfg.colors["blue_muted_lighter_1"],
            cfg.colors["blue_muted_lighter_2"],
        ],  # muted dark blue
        "shd": [
            cfg.colors["red_muted"],
            cfg.colors["red_muted_lighter_1"],
            cfg.colors["red_muted_lighter_2"],
        ],  # muted dark red
        "unq1": [
            cfg.colors["green_muted"],
            cfg.colors["green_muted_lighter_1"],
            cfg.colors["green_muted_lighter_2"],
        ],  # muted dark green
    }
    COLORS["unq2"] = COLORS["unq1"]
    try:
        return COLORS[measure.split("_")[0]]
    except KeyError:
        return [
            cfg.colors["grey_muted"],
            cfg.colors["grey_muted_lighter_1"],
            cfg.colors["grey_muted_lighter_2"],
        ]


def load_experiment_results(
    loadpath, target_variable, folder_steps=5, max_folder=25, load_local_pid=False
):
    """Load experiment results and add them to a single dict

    Parameters
    ----------
    loadpath : pathlib.Path
        Folder containing results from PID estimation
    target_variable : str
        Target variable for synergy estimation
    folder_steps : int, optional
        Number of heuristics used in experiment, by default 5
    max_folder : int, optional
        Maximum number of folders, by default 25
    load_local_pid : bool, optional
        Whether to also load local PID estimates

    Returns
    -------
    dict
        Collected results with structure
        result[coop_parameter][measure][heuristic]
    """
    results = {}
    coop_parameters = []
    folders = np.array([1, folder_steps])
    logging.info("Using target variable %s", target_variable)
    logging.info("Loading data from %s", loadpath)
    i = 0
    while folders[1] < (max_folder + 1):
        filename = f"mi_estimates_over_trials_{folders[0]}_to_{folders[-1]}_t_{target_variable}.p"
        logging.info(  # pylint: disable=W1201
            "Reading results for folders %s to %s: %s"  # pylint: disable=C0209
            % (folders[0], folders[1], filename)
        )
        with open(loadpath.joinpath(filename), "rb") as f:
            dat = pickle.load(f)
        coop_parameters.append(float(dat["experiment_settings"]["environment"]["coop"]))
        results[coop_parameters[-1]] = dat
        if load_local_pid:
            # Struct: c - measure - heuristic
            c = coop_parameters[-1]
            heuristics = list(results[c][next(iter(results[c]))].keys())
            n_trials = len(results[c][next(iter(results[c]))][heuristics[0]])
            results[c]["local_sx_pid"] = {}
            for h in heuristics:
                results[c]["local_sx_pid"][h] = []
                for trial in range(n_trials):
                    filename = f"local_sx_pid_{h}_c_{coop_parameters[-1]:.2f}_t_{target_variable}_trial_{trial}.csv"
                    results[c]["local_sx_pid"][h].append(
                        pd.read_csv(
                            loadpath.joinpath("local_sx_pid", filename), index_col=0
                        )
                    )

        i += 1
        folders[0] = folders[1] + 1
        folders[1] = folders[1] + folder_steps
    return results, coop_parameters


def main(path, target_variable, measures, n_folders, render):
    initialize_logger(log_name=f"compare_measure_by_c_t_{target_variable}")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results, coop_parameters = load_experiment_results(
        loadpath, target_variable, max_folder=n_folders
    )

    fig, ax = plt.subplots(ncols=2, nrows=len(measures), figsize=(5.5, len(measures) * 1.8))
    plot_label_types = get_plot_label(n_labels=(2 * len(measures)))
    if len(measures) == 1:
        ax = ax[np.newaxis, :]
        plot_label_types = plot_label_types[np.newaxis, :]
    else:
        plot_label_types = plot_label_types.reshape(len(measures), 2)
    for a, l, measure in zip(ax, plot_label_types, measures):
        plot_measure_lineplot(results, coop_parameters, measure, a, l)
        unify_axis_ylim(a)

    if not np.any([True for s in measures if "coop_ratio" in s]):
        logging.info("Unifying axis limits")
        unify_axis_ylim(a)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    measure_label = (
        "_".join([m.split("_")[0] for m in measures])
        + "_"
        + "_".join(measures[0].split("_")[1:])
    )
    filename = loadpath.joinpath(
        f"measure_{measure_label}_by_c_t_{target_variable}.{FILETYPE}"
    )
    logging.info("Saving figure to %s", filename)
    fig.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def unify_axis_ylim(ax):
    # Alternative solution to common y-axis, since sharey=True doesn't show
    # axis labels, fix as shown here:
    # https://stackoverflow.com/questions/29266966/show-tick-labels-when-sharing-an-axis
    # is also not working.
    ylim = [np.inf, -np.inf]
    for a in ax.flatten():
        ylim[0] = np.min([ylim[0], a.get_ylim()[0]])
        ylim[1] = np.max([ylim[1], a.get_ylim()[1]])
    plt.setp(ax, ylim=ylim)


def unify_axis_xlim(ax):
    # Alternative solution to common y-axis, since sharey=True doesn't show
    # axis labels, fix as shown here:
    # https://stackoverflow.com/questions/29266966/show-tick-labels-when-sharing-an-axis
    # is also not working.
    xlim = [np.inf, -np.inf]
    for a in ax.flatten():
        xlim[0] = np.min([xlim[0], a.get_xlim()[0]])
        xlim[1] = np.max([xlim[1], a.get_xlim()[1]])
    plt.setp(ax, xlim=xlim)


def set_plotlabel(ax, axlabel, x_pos=-0.15, y_pos=1.01):
    ax.text(
        x_pos, y_pos, axlabel, transform=ax.transAxes, fontsize=12, fontweight="bold"
    )


def get_mean_pid(results, coop_parameters, measure):
    c_temp = list(results.keys())[0]
    try:
        means = pd.DataFrame(
            columns=list(results[c_temp][measure].keys()),
            index=coop_parameters,
        )
    except KeyError as exc:
        raise KeyError(
            "Unknown measure, try one of %s", list(results[c_temp].keys())
        ) from exc
    stds = pd.DataFrame(
        columns=list(results[c_temp][measure].keys()), index=coop_parameters
    )

    for coop in coop_parameters:
        means.loc[coop] = pd.DataFrame(results[coop][measure]).mean()
        stds.loc[coop] = pd.DataFrame(results[coop][measure]).std(ddof=1)
    heuristic_labels = {c: cfg.labels[c] for c in results[c_temp][measure].keys()}
    means.rename(columns=heuristic_labels, inplace=True)
    stds.rename(columns=heuristic_labels, inplace=True)

    return means, stds


def plot_measure_lineplot(results, coop_parameters, measure, ax, ax_label):
    sns.set_style("ticks")
    means, stds = get_mean_pid(results, coop_parameters, measure)

    def _plot_lines(df, colors, ax, axlabel):
        styles = ["-.", "--", ":"]
        for col, style, c in zip(
            df.columns, styles, colors
        ):  # loop over cols to set style and color
            df[col].plot(
                yerr=stds[col],
                capsize=2,
                capthick=0.5,
                elinewidth=0.5,
                color=c,
                # fmt=style,
                ax=ax,
            ).legend(loc="best")
        set_plotlabel(ax, axlabel)

    _plot_lines(
        means[["BL", "Ego"]],
        get_colors(measure),
        ax[0],
        ax_label[0],
    )
    _plot_lines(
        means[["Coop", "Adapt", "Social"]],
        get_colors(measure),
        ax[1],
        ax_label[1],
    )

    for a in ax:
        a.set(xlabel="$c$", ylabel=cfg.labels[measure])
        a.axhline(0, zorder=0, color="lightgray", linestyle=":")
        a.tick_params(axis="y", labelleft=True)
    # plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.grid(which="major", axis="both")
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot estimated measures as a function of environment cooperativity, c"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v13",
        type=str,
        help="Path to analysis settings file",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    parser.add_argument(
        "-m",
        "--measures",
        nargs="+",
        default=["shd_sx, syn_sx"],  # syn_norm_sx, syn_norm_sx_cor, syn_norm_iccs
        help=("Measures to plot"),
    )
    parser.add_argument(
        "-f",
        "--folders",
        default=25,
        type=int,
        help=("Number of results folders to parse"),
    )
    args = parser.parse_args()

    main(args.path, args.target, args.measures, args.folders, args.render)
