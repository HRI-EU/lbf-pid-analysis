#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Plot PID profiles for asymmetric agent strenghts.
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
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils import initialize_logger
from plot_measure_by_c import (
    load_experiment_results,
    get_mean_pid,
    get_colors,
    get_plot_label,
    set_plotlabel,
)
import config as cfg

FILETYPE = "pdf" # "png"
TARGET_VARS = []
MAX_FOLDER = 30
FOLDER_STEPS = 6
COOP_PARAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(path, heuristic, estimator, n_folders, folder_steps, render):
    initialize_logger(log_name="compare_synergy_asymmetric")
    logging.info("Plotting asymmetric results for heuristic %s", heuristic)
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results_any_food, coop_parameters = load_experiment_results(
        loadpath,
        target_variable="any_food",
        folder_steps=folder_steps,
        max_folder=n_folders,
    )
    results_coll_a0, _ = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_0",
        folder_steps=folder_steps,
        max_folder=n_folders,
    )
    results_coll_a1, _ = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_1",
        folder_steps=folder_steps,
        max_folder=n_folders,
    )
    fig = plot_asymmetric_results(
        results_any_food,
        results_coll_a0,
        results_coll_a1,
        coop_parameters,
        estimator,
        heuristic,
    )
    filename = loadpath.joinpath(
        f"asymmetric_results_{heuristic.lower()}_{estimator}.{FILETYPE}"
    )
    logging.info("Saving figure to %s", filename)
    fig.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def plot_asymmetric_results(
    results_any_food,
    results_collections_a0,
    results_collections_a1,
    coop_parameters,
    estimator,
    heuristic,
):
    """Plot PID results for LBF experiment with asymmetric agent levels

    Plot results for asymmetric agent capabilities, i.e., different levels and
    thus different capabilities to collect food items (we have one "stronger"
    and one "weaker" agent).

    Parameters
    ----------
    results_any_food : dict
        PID estimates for target variable any_food (joint goal)
    results_collections_a0 : dict
        PID estimates for target variable n_collection_a0 (goal of first agent
        only)
    results_collections_a1 : dict
        PID estimates for target variable n_collection_a1 (goal of second agent
        only)
    coop_parameters : iterable
        List of cooperation parameters, c, over which to plot PID results
    estimator : str
        Which PID estimates to plot
    heuristic : str
        Which agent heuristic to plot

    Returns
    -------
    matplotlib.figure
        Results figure handle
    """
    sns.set_style("ticks")
    fig, ax = plt.subplots(nrows=3, figsize=(2.8, 5.2))
    results = [results_any_food, results_collections_a0, results_collections_a1]
    plot_labels = get_plot_label(len(results), "A")
    results_labels = [
        cfg.labels["any_food_collected"],
        cfg.labels["n_collections_agent_0"],
        cfg.labels["n_collections_agent_1"],
    ]
    for r, a, p, t in zip(results, ax, plot_labels, results_labels):
        plot_pid_profile_by_c(
            r, heuristic, coop_parameters, estimator, ax=a, ax_label=p
        )
        a.set_title(t, fontsize=8, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_pid_profile_by_c(results, heuristic, coop_parameters, estimator, ax, ax_label):
    """Plot PID profile over values for cooperation parameter c.

    Plot each PID atom (shared, synergy, unique 1, unique 2) as a function of
    the cooperation parameter c.

    Parameters
    ----------
    results : dict
        PID estimates
    heuristic : str
        Agent heuristic for which to plot PID estimates
    coop_parameters : iterable
        List of cooperation parameters for which to plot PID estimates
    estimator : str
        Which PID estimator results to use
    ax : matplotlib.axis
        Axis to plot into
    ax_label : str
        Label to set for axis

    Raises
    ------
    KeyError
        For heuristics not contained in the results dict
    """
    measures = [f"{m}_{estimator}" for m in ["shd", "syn", "unq1", "unq2"]]
    for measure, style in zip(measures, ["-", "-", "--", ":"]):
        means, stds = get_mean_pid(results, coop_parameters, measure)
        try:
            means[heuristic].plot(
                yerr=stds[heuristic],
                capsize=2,
                capthick=0.5,
                elinewidth=0.5,
                color=get_colors(measure),
                fmt=style,
                ax=ax,
            )  # .legend(loc="best")
        except KeyError as err:
            raise KeyError(
                f"Unknown heuristic {heuristic}, available heuristics: {list(means.keys())}"
            ) from err
    ax.legend([cfg.labels_short[m] for m in measures], prop={"size": 6})
    ax.set(xlabel="$c$", ylabel="PID")
    ax.axhline(0, zorder=0, color="lightgray", linestyle=":")
    ax.tick_params(axis="y", labelleft=True)
    set_plotlabel(ax, ax_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot estimated measures for experimental condition with asymmetric agent capabilities"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/asymmetric_d_0_0_v13",
        type=str,
        help="Path to analysis settings file",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        default="Adapt",
        help=("Agent heuristic to plot"),
    )
    parser.add_argument(
        "-e",
        "--estimator",
        type=str,
        default="norm_sx_cor",
        help=("Estimator results to plot"),
    )
    parser.add_argument(
        "-f",
        "--folders",
        default=44,
        type=int,
        help=("Number of results folders to parse"),
    )
    parser.add_argument(
        "-s",
        "--folder_steps",
        default=4,
        type=int,
        help=("Number of results folders in one condition"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.heuristic,
        args.estimator,
        args.folders,
        args.folder_steps,
        args.render,
    )
