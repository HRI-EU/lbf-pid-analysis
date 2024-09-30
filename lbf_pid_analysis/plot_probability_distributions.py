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
import logging
import glob
import argparse
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from utils import initialize_logger, read_settings
from analyze_pid_per_trial import generate_output_path
from plot_measure_by_c import load_experiment_results, unify_axis_ylim
from compare_correlated_logic_gates_pid import generate_empty_dist
import config as cfg

FIGTYPE = "pdf"  # "pdf"/"png"


def main(path, target_variable, n_folders, render):
    outpath = generate_output_path(path)
    results, _ = load_experiment_results(
        outpath, target_variable, max_folder=n_folders, load_local_pid=True
    )
    plot_avg_joint_distributions(
        results,
        outpath,
        filename=f"probability_distributions_t_{target_variable}",
        render=render,
    )


def plot_avg_joint_distributions(results, outpath, filename, render=False):

    coop_params = list(results.keys())
    heuristics = results[coop_params[0]]["local_sx_pid"].keys()
    fig, ax = plt.subplots(
        ncols=len(coop_params), nrows=len(heuristics), figsize=(10, 8), sharey=True
    )

    for i, heuristic in enumerate(heuristics):
        for j, coop_param in enumerate(coop_params):
            realizations = list(product(*[np.arange(2), np.arange(2), np.arange(2)]))
            dist = {r: [] for r in realizations}
            for df in results[coop_param]["local_sx_pid"][heuristic]:

                df["realizations"] = [
                    tuple(r)
                    for r in np.array(
                        [df["s1"].values, df["s2"].values, df["t"].values]
                    ).T
                ]
                for _, r in df.iterrows():
                    dist[r["realizations"]].append(r["p(s1,s2,t)"])

            dist_df = {}
            for r in dist:
                if len(dist[r]) == 0:
                    dist_df[r] = [0, 0]
                else:
                    dist_df[r] = [np.mean(dist[r]), np.std(dist[r], ddof=1)]
            dist_df = pd.DataFrame.from_dict(
                dist_df, orient="index", columns=["mean", "sd"]
            )
            # print(dist_df)
            dist_df.drop([(0, 0, 0)], inplace=True)
            real = [str(r) for r in dist_df.index]
            ax[i, j].bar(x=real, height=dist_df["mean"], color="gray")
            ax[i, j].errorbar(
                x=real,
                y=dist_df["mean"],
                yerr=dist_df["sd"],
                fmt=".",  # plot error markers only, the line is plottled slightly thicker below
                markersize=0,
                capsize=2,
                elinewidth=0.5,
                color="k",
            )
            ax[i, j].set(title=f"{cfg.labels[heuristic]} - c={coop_param}")
            ax[i, j].tick_params(labelrotation=90)
    # unify_axis_ylim(ax)
    plt.tight_layout()
    logging.info("Saving figure to %s", outpath.joinpath(f"{filename}.{FIGTYPE}"))
    plt.savefig(outpath.joinpath(f"{filename}.{FIGTYPE}"))
    if render:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot probability functions used in PID estimation."
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
        "--total_folders",
        default=55,
        type=int,
        help="Total number of folders",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    args = parser.parse_args()

    main(args.path, args.target, args.total_folders, args.render)
