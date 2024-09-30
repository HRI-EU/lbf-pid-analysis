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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from utils import initialize_logger, read_settings
from analyze_pid_per_trial import (
    generate_output_path,
    get_heuristic_used,
)
from plot_measure_by_c import load_experiment_results, unify_axis_ylim
import config as cfg

FIGTYPE = "pdf"  # "pdf"/"png"


def main(path, folder_number, target_variable, n_trials, n_folders, render=False):
    initialize_logger(
        log_name="plot_local_sx_pid_t_{}_folder_{}".format(
            target_variable, folder_number
        ),
    )
    outpath = generate_output_path(path)
    local_pid_inpath = Path(outpath).joinpath("local_sx_pid")
    local_pid_outpath = Path(outpath).joinpath("local_sx_pid_plots")
    local_pid_outpath.mkdir(parents=True, exist_ok=True)

    # Identify files collected for experiment. Load settings file.
    foldername = Path(path, f"{int(folder_number):02d}*")
    experiment_settings = read_settings(
        glob.glob(str(foldername.joinpath("experiment_settings.yml")))[0]
    )
    heuristic = get_heuristic_used(experiment_settings)
    coop_param = float(experiment_settings["environment"]["coop"])

    # Print local stats.
    results, _ = load_experiment_results(
        outpath, target_variable, max_folder=n_folders, load_local_pid=True
    )
    plot_local_pid_stats(
        results[coop_param]["local_sx_pid"][heuristic],
        local_pid_outpath,
        filename=f"local_sx_pid_{heuristic}_c_{coop_param:.2f}_t_{target_variable}",
    )

    # Plot requested number of trials as heat maps
    for trial in range(n_trials):  # experiment_settings["experiment"]["ntrials"]
        logging.info(
            "Plotting trial %d (folder %d, heuristic %s, coop: %.2f)",
            trial,
            folder_number,
            heuristic,
            float(experiment_settings["environment"]["coop"]),
        )
        filename = f"local_sx_pid_{heuristic}_c_{coop_param:.2f}_t_{target_variable}_trial_{trial}"
        df = pd.read_csv(local_pid_inpath.joinpath(f"{filename}.csv"))
        plot_local_sx_pid(
            df,
            figuretitle=f"Local SxPID-T={target_variable}, {cfg.labels[heuristic]}, trial={trial}, c={float(experiment_settings['environment']['coop']):.2f}",
            filename=local_pid_outpath.joinpath(f"{filename}.{FIGTYPE}"),
            render=render,
        )


def plot_local_sx_pid(df, figuretitle, filename, render=False):
    """Plot local Sx PID as heatmaps

    Parameters
    ----------
    df : pandas.DataFrame
        Local SxPID estimates per triplet of realizations
    figuretitle : str
        Figure title
    filename : pathlib.Path
        Figure save path
    render : bool, optional
        Whether to display figure before saving, by default False
    """
    df.sort_values(
        by=[cfg.col_t, cfg.col_s1, cfg.col_s2], ascending=False, inplace=True
    )

    fig_height = df.shape[0] * 0.8
    _, ax = plt.subplots(
        ncols=13,
        figsize=(7.5, fig_height),
        gridspec_kw={"width_ratios": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
    )

    def plot_heatmap(dat, ax, cmap, vmin=None, vmax=None, fmt=".3f", annot_font_size=7):
        sns.heatmap(  # counts
            dat,
            ax=ax,
            cbar=False,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            annot_kws={"size": annot_font_size},
            vmin=vmin,
            vmax=vmax,
        )

    plot_heatmap(  # realizations
        df[[cfg.col_s1, cfg.col_s2, cfg.col_t]],
        ax=ax[0],
        cmap="Pastel1",
        fmt="d",
        vmin=0,
        vmax=5,
    )
    plot_heatmap(df[[cfg.col_count]], ax=ax[1], cmap="Greys", fmt="d")  # counts
    plot_heatmap(  # joint probabilities
        df[[cfg.col_joint_prob]], ax=ax[2], cmap="Greys", vmin=0, vmax=1.0
    )
    plot_heatmap(df[[cfg.col_lmi_s1_s2_t]], ax=ax[3], cmap="Greys")  # local joint MI

    # Local PID estimates
    plot_heatmap(df[[cfg.col_unq_s1]], ax=ax[4], cmap="Greens")
    plot_heatmap(df[[cfg.col_unq_s2]], ax=ax[5], cmap="Greens")
    plot_heatmap(df[[cfg.col_shd]], ax=ax[6], cmap="Reds")
    plot_heatmap(df[[cfg.col_syn]], ax=ax[7], cmap="Blues")

    # Local weighted PID estimates
    plot_heatmap(df[["mi*p"]], ax=ax[8], cmap="Greys")
    plot_heatmap(df[["unq_s1*p"]], ax=ax[9], cmap="Greens")
    plot_heatmap(df[["unq_s2*p"]], ax=ax[10], cmap="Greens")
    plot_heatmap(df[["shd*p"]], ax=ax[11], cmap="Reds")
    plot_heatmap(df[["syn*p"]], ax=ax[12], cmap="Blues")

    plt.suptitle(figuretitle)
    for a in ax.flatten():
        a.xaxis.set_ticks_position("top")
        a.axes.get_yaxis().set_visible(False)
        a.set_xticklabels(a.get_xticklabels(), rotation=45)
        for _, spine in a.spines.items():
            spine.set_visible(True)

    logging.info("Local SxPID\n %s", df)
    logging.info("Summed, weighted synergy: %.4f", df["syn*p"].sum())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    print(f"Saving figure to {filename}")
    plt.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()
    return df


def plot_local_pid_stats(local_pid, outpath, filename, weighted=True):
    for df in local_pid:
        df.sort_values(
            by=[cfg.col_t, cfg.col_s1, cfg.col_s2], ascending=False, inplace=True
        )
        df["realizations"] = [
            tuple(r)
            for r in np.array([df["s1"].values, df["s2"].values, df["t"].values]).T
        ]
        df.set_index("realizations", inplace=True)
        df.drop(columns=["s1", "s2", "t"], inplace=True)
    df = pd.concat(local_pid, keys=np.arange(len(local_pid)))
    df_counts = pd.DataFrame(
        [df.xs(col, axis=1).unstack().count() for col in df.columns], index=df.columns
    ).T
    df_means = pd.DataFrame(
        [df.xs(col, axis=1).unstack().mean() for col in df.columns], index=df.columns
    ).T
    df_sds = pd.DataFrame(
        [df.xs(col, axis=1).unstack().std(ddof=1) for col in df.columns],
        index=df.columns,
    ).T.fillna(0)

    logging.info(
        "Statistics for local PID estimates over all trials:\nMeans:\n%s\nSD:\n%s\nCounts\n%s",
        df_means,
        df_sds,
        df_counts,
    )
    logging.info("Saving statistics to %s/%s*.csv", outpath, filename)
    df_means.to_csv(outpath.joinpath(f"{filename}_means.csv"))
    df_sds.to_csv(outpath.joinpath(f"{filename}_sds.csv"))
    df_counts.to_csv(outpath.joinpath(f"{filename}_counts.csv"))

    plot_columns = ["p(s1,s2,t)"]
    if weighted:
        plot_columns += ["mi*p", "unq_s1*p", "unq_s2*p", "shd*p", "syn*p"]
    else:
        plot_columns += ["i(s1,s2;t)", "unq_s1", "unq_s2", "shd", "syn"]
    colors = ["lightgray"] + [
        "gray",
        cfg.colors["green_muted_lighter_1"],
        cfg.colors["green_muted_lighter_1"],
        cfg.colors["red_muted_lighter_1"],
        cfg.colors["blue_muted_lighter_1"],
    ]
    fig = plt.figure(figsize=(len(plot_columns) * 0.6, 3))
    gs = GridSpec(len(df_means), 2, width_ratios=[1, len(plot_columns)])
    ax_pid = []
    ax_prob = []
    for i in np.arange(len(df_means)):
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax1.set(ylabel=df_means.index[i])
        for a, column, color in zip(
            [ax1, ax2], [[plot_columns[0]], plot_columns[1:]], [colors[0], colors[1:]]
        ):
            a.bar(x=column, height=df_means[column].iloc[i], color=color)
            a.errorbar(
                x=column,
                y=df_means[column].iloc[i],
                yerr=df_sds[column].iloc[i],
                fmt=".",  # plot error markers only, the line is plottled slightly thicker below
                markersize=0,
                capsize=2,
                elinewidth=0.5,
                color="k",
            )
            a.axhline(0, lw=0.7, color="k")
        ax_prob.append(ax1)
        ax_pid.append(ax2)
        # for j, col in zip(np.arange(len(plot_columns)), plot_columns):
        #     ax[i, j].bar(x=[col], height=df_means[col].iloc[i], color=colors[j])
        #     ax[i, j].errorbar(
        #         x=[col],
        #         y=df_means[col].iloc[i],
        #         yerr=df_sds[col].iloc[i],
        #         fmt="",  # plot error markers only, the line is plottled slightly thicker below
        #         capsize=2,
        #         elinewidth=0.5,
        #         color="k",
        #     )
        #     ax[i, j].axhline(0, lw=0.7, color="k")
    unify_axis_ylim(np.array(ax_prob))
    unify_axis_ylim(np.array(ax_pid))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath.joinpath(f"{filename}_all_trials_stats.{FIGTYPE}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Information-theoretic analysis of LBF experiments."
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
        "-f",
        "--folder",
        default=1,
        type=int,
        help="Number of the folder to analyze, corresponds to one heuristic",
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
    parser.add_argument(
        "--trials",
        default=3,
        type=int,
        help=("N first trials for which to plot local PID"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.folder,
        args.target,
        args.trials,
        args.total_folders,
        args.render,
    )
