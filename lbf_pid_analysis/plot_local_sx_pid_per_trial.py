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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import initialize_logger, read_settings
from analyze_pid_per_trial import (
    load_experiment_data,
    generate_output_path,
    get_trial_file_names,
    get_heuristic_used,
    get_experiment_performance,
    encode_source_variables,
    encode_target_variable,
)
from pid_estimation import estimate_sx_pid
import config as cfg


def main(
    path, settings_path, folder_number, trial, target_variable, render=False
):
    initialize_logger(
        log_name="plot_local_sx_pid_t_{}_folder_{}_trial_{}".format(
            target_variable, folder_number, trial
        ),
    )
    outpath = generate_output_path(path)
    local_pid_outpath = Path(outpath).joinpath("local_sx_pid")
    local_pid_outpath.mkdir(parents=True, exist_ok=True)

    # Identify files collected for experiment. Load settings file.
    foldername = Path(path, f"{int(folder_number):02d}*")
    trial_files = get_trial_file_names(foldername)
    experiment_settings = read_settings(
        glob.glob(str(foldername.joinpath("experiment_settings.yml")))[0]
    )
    analysis_settings = read_settings(Path(settings_path))

    # Create data structures to collect data over trials.
    heuristic = get_heuristic_used(experiment_settings)
    logging.info(
        "Analyzing trial %d (folder %d, heuristic %s, coop: %.2f)",
        trial,
        folder_number,
        heuristic,
        float(experiment_settings["environment"]["coop"]),
    )

    # Load experiment data, movements, field setup, performance.
    experiment_data = load_experiment_data(trial_files[trial])
    performance = get_experiment_performance(
        experiment_data, experiment_settings, trial, outpath=None, render=False
    )

    # Encode variables for information-theoretic analysis and estimate PID.
    source_0, source_1 = encode_source_variables(
        experiment_data,
        source_type=analysis_settings["sources"],
        source_encoding=analysis_settings["source_encoding"],
    )
    target, _ = encode_target_variable(performance, target_variable)
    if np.all(source_0 == source_0[0]) or np.all(source_1 == source_1[0]):
        logging.info("No agent movement for this trial")
    pid_sx = estimate_sx_pid(source_0, source_1, target, correction=False)

    filename = f"local_sx_pid_{heuristic}_c_{float(experiment_settings['environment']['coop']):.2f}_t_{target_variable}_trial_{trial}"
    df = plot_local_sx_pid(
        pid_sx["local"],
        target_variable,
        filename=local_pid_outpath.joinpath(f"{filename}.pdf"),
        render=render,
    )
    df.to_csv(local_pid_outpath.joinpath(f"{filename}.csv"))


def plot_local_sx_pid(df, target_variable, filename, render=False):
    """Plot local Sx PID as heatmaps

    Parameters
    ----------
    df : pandas.DataFrame
        Local SxPID estimates per triplet of realizations
    target_variable : str
        Target variable used for PID estimation
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
        ncols=10,
        figsize=(7, fig_height),
        gridspec_kw={"width_ratios": [2, 1, 1, 2, 1, 1, 1, 1, 1, 1]},
    )

    # Plot outcomes
    sns.heatmap(
        df[[cfg.col_s1, cfg.col_s2, cfg.col_t]],
        ax=ax[0],
        cbar=False,
        cmap="Pastel1",
        # annot=df[['source1'_LABEL, 'source2'_LABEL, 'target'_LABEL]], annot_kws={"size": annot_font_size}, fmt='',
        annot=True,
        vmin=0,
        vmax=5,
    )
    fmt_pid = ".3f"
    annot_font_size = 7
    # Plot probabilities of individual outcomes
    sns.heatmap(
        df[[cfg.col_count]],
        ax=ax[1],
        cbar=False,
        cmap="Greys",
        annot=True,
        fmt="d",
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[[cfg.col_joint_prob]],
        ax=ax[2],
        cbar=False,
        cmap="Greys",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
        vmin=0,
        vmax=1.0,
    )
    sns.heatmap(
        df[[cfg.col_lmi_s1_s2_t, cfg.col_lmi_s1_t, cfg.col_lmi_s2_t]],
        ax=ax[3],
        cbar=False,
        cmap="Greys",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )

    # Plot local PID estimates
    df["syn/lmi"] = df[cfg.col_syn] / df[cfg.col_lmi_s1_s2_t]
    df["syn*p"] = df[cfg.col_syn] * df[cfg.col_joint_prob]
    sns.heatmap(
        df[[cfg.col_unq_s1]],
        ax=ax[4],
        cbar=False,
        cmap="Greens",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[[cfg.col_unq_s2]],
        ax=ax[5],
        cbar=False,
        cmap="Greens",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[[cfg.col_shd]],
        ax=ax[6],
        cbar=False,
        cmap="Reds",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[[cfg.col_syn]],
        ax=ax[7],
        cbar=False,
        cmap="Blues",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[["syn/lmi"]],
        ax=ax[8],
        cbar=False,
        cmap="Blues",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    sns.heatmap(
        df[["syn*p"]],
        ax=ax[9],
        cbar=False,
        cmap="Blues",
        annot=True,
        fmt=fmt_pid,
        annot_kws={"size": annot_font_size},
    )
    plt.suptitle(f"SxPID - target {target_variable}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Information-theoretic analysis of LBF experiments."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v7",
        type=str,
        help="Path to experimental results",
    )
    parser.add_argument(
        "-s",
        "--settings",
        default="../settings/analysis_settings.yml",
        type=str,
        help="Path to analysis settings file",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default=1,
        type=int,
        help="Number of the folder to analyze, corresponds to one heuristic",
    )
    parser.add_argument(
        "--trial",
        default=1,
        type=int,
        help="Trial to plot from specified folder",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.settings,
        args.folder,
        args.trial,
        args.target,
        args.render,
    )
