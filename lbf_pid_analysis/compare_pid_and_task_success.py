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
import seaborn as sns

from utils import initialize_logger
from plot_measure_by_c import load_experiment_results, set_layout, unify_axis_ylim
import config as cfg

FILETYPE = "pdf" # "png"

def main(
    path, target_variable, estimator, measure_task_success, coop_plot, n_folders, render
):
    initialize_logger(log_name=f"compare_measure_by_c_t_{target_variable}")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results, _ = load_experiment_results(
        loadpath, target_variable, max_folder=n_folders
    )
    compare_pid_and_task_success(
        results,
        estimator,
        measure_task_success,
        coop_plot,
        filename=loadpath.joinpath(
            f"comparison_pid_{estimator}_task_success_{measure_task_success}_c_{str(coop_plot).replace('.', '_')}.{FILETYPE}"
        ),
        render=render,
    )


def compare_pid_and_task_success(
    results,
    estimator,
    measure_task_success,
    coop_plot,
    filename,
    heuristics_used=None,
    render=False,
):
    if heuristics_used is None:
        heuristics_used = [
            "MH_EGOISTIC",
            "MH_SOCIAL1",
            "MH_ADAPTIVE",
            "MH_COOPERATIVE",
        ]
    set_layout()
    try:
        results[coop_plot]
    except KeyError as exc:
        raise KeyError(
            "Unknown cooperation parameter {}, available parameters: {}".format(
                coop_plot,
                list(results.keys()),
            )
        ) from exc
    try:
        results[0.0][f"syn_{estimator}"]
    except KeyError as exc:
        raise KeyError(
            "Unknown measure/estimator %s, available measures: %s",
            estimator,
            list(results[0.0].keys()),
        ) from exc

    fig, ax = plt.subplots(
        ncols=len(heuristics_used) + 2,
        # figsize=(
        #     (len(heuristics_used) + 1)
        #     * cfg.plot_elements["figsize"]["colwidth_in"]  # pylint: disable=no-member
        #     * 1.2,
        #     1.2
        #     * cfg.plot_elements["figsize"][  # pylint: disable=no-member
        #         "lineheight_in"
        #     ],
        # ),
        figsize=(10, 2),
    )

    # Plot task success and MI
    if estimator.split("_")[0] == "norm":
        mi_measure = f"mi_{estimator[5:]}"
    else:
        mi_measure = f"mi_{estimator}"
    for measure, a in zip([measure_task_success, mi_measure], ax[:2]):
        try:
            df = pd.DataFrame(results[coop_plot][measure])[heuristics_used]
        except KeyError as exc:
            raise KeyError(
                "Available measures: {}".format(list(results[coop_plot].keys())),
            ) from exc
        sns.boxplot(
            x="heuristic",
            y=measure,
            data=df.melt(var_name="heuristic", value_name=measure),
            ax=a,
            color=cfg.colors["gray"],
            # hue="measure",
            width=0.6,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        a.set(xlabel="")
        a.xaxis.set_ticks(np.arange(len(heuristics_used)))
        a.set_xticklabels(
            [cfg.labels[h] for h in heuristics_used],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    pid_atoms = [f"{a}_{estimator}" for a in ["shd", "syn", "unq1", "unq2"]]
    box_color = [
        cfg.colors["red_muted_lighter_1"],
        cfg.colors["blue_muted_lighter_1"],
        cfg.colors["green_muted_lighter_1"],
        cfg.colors["green_muted_lighter_1"],
    ]
    for i, heuristic in enumerate(heuristics_used):

        df_pid = pd.DataFrame(
            [results[coop_plot][a][heuristic] for a in pid_atoms], index=pid_atoms
        ).T
        sns.boxplot(
            x="pid_atoms",
            y=heuristic,
            data=df_pid.melt(var_name="pid_atoms", value_name=heuristic),
            ax=ax[i + 2],
            palette=box_color,
            # hue="measure",
            width=0.6,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        ax[i + 2].set(ylabel=cfg.labels[heuristic], xlabel="")

    atom_labels = [cfg.labels_short[a] for a in pid_atoms]
    for a in ax[2:]:
        a.xaxis.set_ticks(
            np.arange(len(atom_labels))
        )  # to avoid FixedFormatter warning
        a.set_xticklabels(
            atom_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    unify_axis_ylim(ax[2:])

    fig.suptitle(f"c={coop_plot}")
    fig.subplots_adjust(left=0.07, bottom=0.27, wspace=0.8, right=0.98)
    logging.info("Saving figure to %s", filename)
    plt.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def _get_pid_atoms(estimator, atoms=None):
    if atoms is None:
        atoms = ["shd", "syn", "unq1", "unq2", "mi"]
    estimator_atoms = [f"{a}_{estimator}" for a in atoms[:-1]]
    if estimator.split("_")[0] == "norm":
        return estimator_atoms + [f"{atoms[-1]}_{estimator[5:]}"]
    return estimator_atoms + [f"{atoms[-1]}_{estimator}"]


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
        "-e",
        "--estimator",
        type=str,
        default="norm_sx_cor",  # syn_norm_sx, syn_norm_sx_cor, syn_norm_iccs
        help=("Estimator results to plot"),
    )
    parser.add_argument(
        "-s",
        "--success",
        default="total_food_value_collected",
        type=str,
        help=("Variable to use as measure of task success"),
    )
    parser.add_argument(
        "-c",
        default="0.8",
        type=float,
        help=("Cooperation parameter for which to compare"),
    )
    parser.add_argument(
        "-f",
        "--folders",
        default=25,
        type=int,
        help=("Number of results folders to parse"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.target,
        args.estimator,
        args.success,
        args.c,
        args.folders,
        args.render,
    )
