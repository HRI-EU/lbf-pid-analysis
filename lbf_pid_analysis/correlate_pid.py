#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contrast PID profiles and total task success to show that PID can
# differentiate between different strategies that lead to similar results.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import statsmodels.api as sm

from utils import initialize_logger
from plot_measure_by_c import (
    load_experiment_results,
    unify_axis_ylim,
    unify_axis_xlim,
)
import config as cfg


FILETYPE = "pdf" # "png"


def main(path, target_variable, x, y, n_folders, render):
    initialize_logger(log_name=f"correlate_{x}_with_{y}")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results, coop_parameters = load_experiment_results(
        loadpath, target_variable, max_folder=n_folders, load_local_pid=True
    )
    correlate_pid_and_task_success(
        results,
        x,
        y,
        coop_parameters,
        filename=loadpath.joinpath(f"correlation_{x}-{y}.{FILETYPE}"),
        render=render,
    )


def correlate_pid_and_task_success(
    results,
    x,
    y,
    coop_parameters,
    filename,
    heuristics_used=None,
    render=False,
):
    results = collect_local_pid_measures(results)
    if heuristics_used is None:
        heuristics_used = [
            "MH_EGOISTIC",
            "MH_SOCIAL1",
            "MH_ADAPTIVE",
            "MH_COOPERATIVE",
        ]
    SMALL = 8
    # MEDIUM = 12
    # LARGE = 14
    plt.rc("axes", labelsize=SMALL)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
    try:
        results[0.0][x]
    except KeyError as exc:
        raise KeyError(
            f"Unknown variable {x}, available measures: {list(results[0.0].keys())}",
        ) from exc
    try:
        results[0.0][y]
    except KeyError as exc:
        raise KeyError(
            f"Unknown variable {y}, available measures: {list(results[0.0].keys())}",
        ) from exc
    logging.info("Correlating %s with %s", x, y)
    fig, ax = plt.subplots(
        ncols=len(coop_parameters),
        nrows=len(heuristics_used),
        figsize=(15, 7),
    )
    color = cfg.colors["red_muted_lighter_1"]
    for i, heuristic in enumerate(heuristics_used):
        for j, coop in enumerate(coop_parameters):

            df_plot = pd.DataFrame(
                {
                    x: results[coop][x][heuristic],
                    y: results[coop][y][heuristic],
                }
            )
            df_plot.plot.scatter(x=x, y=y, ax=ax[i, j], color=color)
            plot_regression_line(
                df_plot[x],
                df_plot[y],
                ax=ax[i, j],
                linestyle="k-",
            )
            ax[i, j].set(ylabel="", xlabel="")
            ax[i, j].set_title(
                f"{cfg.labels[heuristic]} c={coop}", fontsize=8, fontweight="bold"
            )
            ax[i, j].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[len(heuristics_used) - 1, 0].set(
        xlabel=cfg.labels_short[x], ylabel=cfg.labels_short[y]
    )

    for a in ax:
        unify_axis_xlim(a)
        unify_axis_ylim(a)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.45, hspace=0.53)
    logging.info("Saving figure to %s", filename)
    plt.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def collect_local_pid_measures(results):

    def get_joint_action(local_pid):
        # Return the joint prob and local shared for joint actions, (1,1,1)
        mask = np.logical_and(
            np.logical_and(local_pid["s1"] == 1, local_pid["s2"] == 1),
            local_pid["t"] == 1,
        )
        i_shd = local_pid.loc[mask, "shd*p"].values
        p = local_pid.loc[mask, cfg.col_joint_prob].values
        if len(i_shd) == 0:
            logging.debug(
                "No joint actions found for %s, c=%.2f", heuristic, coop
            )
            return 0.0, 0.0
        else:
            assert (
                len(i_shd) == 1
            ), f"Multiple values for joint actions: {p}/{i_shd} in\n{local_pid}"
        return i_shd[0], p[0]

    def get_indiv1_action(local_pid):
        # Return the joint prob and local unq1 for individual actions, (1,0,1)
        mask = np.logical_and(
            np.logical_and(local_pid["s1"] == 1, local_pid["s2"] == 0),
            local_pid["t"] == 1,
        )
        i_unq1 = local_pid.loc[mask, "unq_s1*p"].values
        p = local_pid.loc[mask, cfg.col_joint_prob].values
        if len(i_unq1) == 0:
            logging.debug(
                "No indiv1 actions found for %s, c=%.2f", heuristic, coop
            )
            return 0.0, 0.0
        else:
            assert (
                len(i_unq1) == 1
            ), f"Multiple values for indiv1 actions: {p}/{i_unq1} in\n{local_pid}"
        return i_unq1[0], p[0]

    # Struct: c - measure - heuristic
    for coop in results:
        heuristics = list(results[coop][next(iter(results[coop]))].keys())
        results[coop]["local_shd_joint_action"] = {}
        results[coop]["p_joint_action"] = {}
        results[coop]["local_unq1_indiv_action"] = {}
        results[coop]["p_indiv1_action"] = {}
        for heuristic in heuristics:
            results[coop]["local_shd_joint_action"][heuristic] = []
            results[coop]["p_joint_action"][heuristic] = []
            results[coop]["local_unq1_indiv_action"][heuristic] = []
            results[coop]["p_indiv1_action"][heuristic] = []
            for local_pid in results[coop]["local_sx_pid"][heuristic]:
                i_shd, p_shd = get_joint_action(local_pid)
                i_unq1, p_unq1 = get_indiv1_action(local_pid)
                results[coop]["local_shd_joint_action"][heuristic].append(i_shd)
                results[coop]["p_joint_action"][heuristic].append(p_shd)
                results[coop]["local_unq1_indiv_action"][heuristic].append(i_unq1)
                results[coop]["p_indiv1_action"][heuristic].append(p_unq1)
    return results


def plot_regression_line(X, Y, ax, linestyle="k-", fontsize=8):
    if len(np.unique(X)) == 1 or len(np.unique(Y)) == 1:
        logging.info("Constant variable, not plotting a regression line")
        return
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    # print(results.summary())
    X_plot = np.linspace(np.min(X), np.max(X), 100)
    ax.plot(X_plot, X_plot * results.params[1] + results.params[0], linestyle)
    ax.text(
        x=0.07,
        y=0.95,
        s=f"$R^2=${results.rsquared:.2f}\n$\\beta_1=${results.params[1]:.2f}\n($p=${results.pvalues[1]:.4f})",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=8,
        transform=ax.transAxes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot correlation between estimated synergy and task success measured by collected food items/value."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v13",
        type=str,
        help="Path to experimental results",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    parser.add_argument(
        "-x",
        type=str,
        default="shd_norm_sx_cor",  # syn_norm_sx, syn_norm_sx_cor, syn_norm_iccs
        help=("First estimate/variable used for correlation"),
    )
    parser.add_argument(
        "-y",
        default="any_food_collected",
        type=str,
        help=("Second estimate/variable used for correlation"),
    )

    parser.add_argument(
        "-f",
        "--folders",
        default=55,
        type=int,
        help=("Number of results folders to parse"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.target,
        args.x,
        args.y,
        args.folders,
        args.render,
    )
