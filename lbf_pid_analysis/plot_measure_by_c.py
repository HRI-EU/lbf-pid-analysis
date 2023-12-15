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

from utils import initialize_logger, read_settings
import config as cfg

COOP_PARAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
PLOT_BOX_PLOT_OUTLIERS = False


def set_layout():
    """Set matplotlib layout"""
    plt.rc(
        "font", size=cfg.plot_elements["textsize"]["medium"]
    )  # controls default text sizes
    plt.rc(
        "axes", titlesize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the x and y labels
    plt.rc(
        "xtick", labelsize=cfg.plot_elements["textsize"]["small"]
    )  # fontsize of the tick labels
    plt.rc(
        "ytick", labelsize=cfg.plot_elements["textsize"]["small"]
    )  # fontsize of the tick labels
    plt.rc(
        "legend", fontsize=cfg.plot_elements["textsize"]["medium"]
    )  # legend fontsize
    plt.rc(
        "figure", titlesize=cfg.plot_elements["textsize"]["large"]
    )  # fontsize of the figure title


def load_experiment_results(
    loadpath, target_variable, folder_steps=5, max_folder=25
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

    Returns
    -------
    dict
        Collected results with structure
        result[coop_parameter][measure][heuristic]
    """
    results = {}
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
            results[COOP_PARAMS[i]] = pickle.load(f)
        i += 1
        folders[0] = folders[1] + 1
        folders[1] = folders[1] + folder_steps
    return results


def main(path, target_variable, measure, render):
    initialize_logger(log_name=f"compare_measure_by_c_t_{target_variable}")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results = load_experiment_results(loadpath, target_variable)
    plot_measure(
        results,
        measure,
        filename=loadpath.joinpath(
            f"measure_{measure}_by_c_t_{target_variable}.pdf"
        ),
        render=render,
    )


def plot_measure(results, measure, filename, render=False):
    set_layout()
    try:
        heuristics_used = list(results[0.0][measure].keys())
    except KeyError as e:
        logging.info(
            "Unknown measure %s, available measures: %s",
            measure,
            list(results[0.0].keys()),
        )
        raise e
    fig, ax = plt.subplots(
        ncols=len(COOP_PARAMS),
        figsize=(
            cfg.figsize["maxwidth_in"],  # pylint: disable=no-member
            cfg.figsize["lineheight_in"],  # pylint: disable=no-member
        ),
        sharey=True,
    )
    for i, coop in enumerate(COOP_PARAMS):
        sns.boxplot(
            data=pd.DataFrame(results[coop][measure]),
            ax=ax[i],
            color="w",
            showfliers=PLOT_BOX_PLOT_OUTLIERS,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        if PLOT_BOX_PLOT_OUTLIERS:
            sns.stripplot(
                data=pd.DataFrame(results[coop][measure]),
                ax=ax[i],
                color=cfg.colors["bluegray_4"],
                size=cfg.plot_elements["marker_size"],
            )
        # if Y_LIM is not None:
        #     ax.set(ylim=yl)

        ax[i].set(title=f"c={coop}")
        ax[i].tick_params(direction="out", length=2)
    ax[0].set(ylabel=cfg.labels[measure])

    ylim = [np.inf, -np.inf]
    for a in ax:
        ylim[0] = np.min([a.get_ylim()[0], ylim[0]])
        ylim[1] = np.max([a.get_ylim()[1], ylim[1]])
    plt.setp(ax, ylim=ylim)

    heuristic_labels = [cfg.labels[c] for c in heuristics_used]
    sns.despine(offset=2, trim=True)
    for a in ax.flatten():
        a.xaxis.set_ticks(
            np.arange(len(heuristic_labels))
        )  # to avoid FixedFormatter warning
        a.set_xticklabels(
            heuristic_labels,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )

    plt.tight_layout()
    logging.info("Saving figure to %s", filename)
    plt.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot estimated measures as a function of environment cooperativity, c"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v6",
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
        "--measure",
        default="syn_norm_sx_cor",  # syn_norm_sx, syn_norm_sx_cor, syn_norm_iccs
        type=str,
        help=("Measure to plot"),
    )
    args = parser.parse_args()

    main(args.path, args.target, args.measure, args.render)
