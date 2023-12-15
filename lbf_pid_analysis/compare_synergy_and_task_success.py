#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contrast synergy and task success to show that those can diverge.
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
from plot_measure_by_c import load_experiment_results, set_layout
import config as cfg

TARGET_VARS = []
MAX_FOLDER = 30
FOLDER_STEPS = 6
COOP_PARAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(path, target_variable, measure_task_success, render):
    initialize_logger(log_name=f"compare_correlation_t_{target_variable}")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results = load_experiment_results(loadpath, target_variable)
    compare_syn_and_task_success(
        results,
        measure_task_success,
        filename=loadpath.joinpath(
            f"correlation_syn_task_success_t_{target_variable}.pdf"
        ),
        render=render,
    )


def compare_syn_and_task_success(
    results,
    measure_task_success,
    filename,
    heuristics_used=None,
    coop_values=None,
    render=False,
):
    if coop_values is None:
        coop_values = COOP_PARAMS
    measure_syn = "syn_norm_sx_cor"  # 'syn', 'syn_norm'
    # heuristics_used = list(results[0.0][measure_syn].keys())
    if heuristics_used is None:
        heuristics_used = ["MH_COOPERATIVE", "MH_ADAPTIVE"]
    set_layout()
    try:
        results[0.0][measure_syn]
    except KeyError as e:
        logging.info(
            "Unknown measure %s, available measures: %s",
            measure_syn,
            list(results[0.0].keys()),
        )
        raise e

    fig, ax = plt.subplots(
        ncols=len(coop_values),
        figsize=(
            len(coop_values)
            * cfg.plot_elements["figsize"][  # pylint: disable=no-member
                "colwidth_in"
            ]
            * 1.2,
            1.2
            * cfg.plot_elements["figsize"][  # pylint: disable=no-member
                "lineheight_in"
            ],
        ),
        sharey=True,
    )
    for i, coop in enumerate(coop_values):
        try:
            df1 = pd.DataFrame(results[coop][measure_task_success])[
                heuristics_used
            ].melt(var_name="heuristic", value_name=measure_task_success)
        except KeyError as e:
            print(f"Available measures: {results[coop].keys()}")
            raise e
        df2 = pd.DataFrame(results[coop][measure_syn])[heuristics_used].melt(
            var_name="heuristic", value_name=measure_syn
        )
        df1["measure"] = measure_task_success
        df2["measure"] = measure_syn
        df1.rename(columns={measure_task_success: "value"}, inplace=True)
        df2.rename(columns={measure_syn: "value"}, inplace=True)
        df1["value"] = (df1["value"] - df1["value"].mean()) / df1["value"].std()
        df2["value"] = (df2["value"] - df2["value"].mean()) / df2["value"].std()

        sns.boxplot(
            x="heuristic",
            y="value",
            data=pd.concat([df1, df2]),
            ax=ax[i],
            color=cfg.colors["gray"],
            hue="measure",
            width=0.6,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        ax[i].get_legend().remove()
        ax[i].set(xlabel="", ylabel="")

        ax[i].set(title=f"c={coop}")
    ax[0].set(
        ylabel="{}\n{}".format(
            cfg.labels[measure_syn],
            cfg.labels[measure_task_success],
        )
    )
    for a in ax.flatten():
        a.tick_params(direction="out", length=2)
    ax[0].legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.35), fancybox=True, ncol=1
    )

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

    # plt.tight_layout()
    fig.subplots_adjust(left=0.24, bottom=0.4, wspace=0.1, right=0.98)
    logging.info("Saving figure to %s", filename)
    plt.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot correlation between estimated synergy and task success measured by collected food items/value."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/",
        type=str,
        help="Path to experimental results",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="cooperative_actions",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    parser.add_argument(
        "-s",
        "--success",
        default="total_food_value_collected",
        type=str,
        help=("Variable to use as measure of task success"),
    )
    args = parser.parse_args()

    main(args.path, args.target, args.success, args.render)
