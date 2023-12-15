#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Plot type of action performed by different agent heuristics over trials.
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
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import initialize_logger, read_settings

from analyze_pid_per_trial import (
    get_trial_file_names,
    get_heuristic_used,
    load_experiment_data,
    generate_output_path,
    get_experiment_performance,
)
from plot_measure_by_c import set_layout
import config as cfg

MAX_FOLDER = 25
FOLDER_STEPS = 5
COOP_PARAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _append_aggregated_results(results, heuristic, performance):
    """Append agent actions for current heuristic, aggregated over trials."""
    results["total_value"][heuristic].append(
        performance["total_food_value_collected"].sum()
    )
    results["n_coop_actions"][heuristic].append(
        performance["cooperative_actions"].sum()
    )
    results["frac_coop_actions"][heuristic].append(
        performance["frac_cooperative_actions"]
    )
    results["n_collections"][heuristic].append(
        performance["any_food_collected"].sum()
    )
    results["n_individual"][heuristic].append(
        performance["any_food_collected"].sum()
        - performance["cooperative_actions"].sum()
    )
    results["n_incident_coop"][heuristic].append(
        (performance["n_collections"] == 2).sum()
        - performance["cooperative_actions"].sum()
    )
    return results


def main(path, n_trials=10):
    """Collect data across experiments and plot action type"""
    initialize_logger(log_name="plot_action_type")
    outpath = generate_output_path(path)

    results_by_heuristic = {
        "total_value": {},
        "n_collections": {},
        "n_coop_actions": {},
        "frac_coop_actions": {},
        "n_individual": {},
        "n_incident_coop": {},
    }

    fig, ax = plt.subplots(
        ncols=len(COOP_PARAMS),
        nrows=len(results_by_heuristic),
        figsize=(len(COOP_PARAMS) * 2, len(results_by_heuristic) * 2),
        sharex=True,
    )
    folders = np.arange(1, FOLDER_STEPS + 1)
    c = 0
    while folders[1] < (MAX_FOLDER + 1):
        for folder_number in folders:  # each folder holds one experiment
            print("\n")

            # Identify files collected for experiment. Load settings file.
            foldername = Path(path, f"{int(folder_number):02d}*")
            trial_files = get_trial_file_names(foldername)
            experiment_settings = read_settings(
                glob.glob(str(foldername.joinpath("experiment_settings.yml")))[
                    0
                ]
            )

            # Create data structures to collect data over trials.
            heuristic = get_heuristic_used(experiment_settings)
            for measure in results_by_heuristic:
                results_by_heuristic[measure][heuristic] = []
            logging.info(  # pylint: disable=W1201
                "Current experiment was run with heuristic %s sight: %d, coop: %.f"  # pylint: disable=C0209
                % (
                    heuristic,
                    experiment_settings.environment["sight"],
                    float(experiment_settings.environment["coop"]),
                )
            )

            for trial in range(n_trials):
                logging.info(  # pylint: disable=W1201
                    "Reading data for trial %d (folder %d, heuristic %s)"  # pylint: disable=C0209
                    % (trial, folder_number, heuristic)
                )

                # Load experiment data, movements, field setup, performance.
                experiment_data = load_experiment_data(trial_files[trial])
                performance = get_experiment_performance(
                    experiment_data,
                    experiment_settings,
                    trial,
                    outpath=None,
                    render=False,
                )
                results_by_heuristic = _append_aggregated_results(
                    results_by_heuristic, heuristic, performance
                )

        plot_action_types(results_by_heuristic, COOP_PARAMS[c], ax=ax[:, c])
        folders = np.arange(folders[-1] + 1, folders[-1] + FOLDER_STEPS + 1)
        c += 1

    filename = outpath.joinpath("action_types.pdf")
    fig.tight_layout()
    logging.info("Saving figure to %s", filename)
    fig.savefig(filename)
    plt.close()


def plot_action_types(results_by_heuristic, coop, ax):
    """Plot type of agent action for current set of experiments.

    Generate box plots for each heuristic. Generate a new subplot for each
    action type in the results.

    Parameters
    ----------
    results_by_heuristic : dict
        Collected action types over experiments
    coop : float
        Cooperation parameter used in current set of experiments.
    ax : iterable
        Set of axes into which to plot
    """
    set_layout()
    folder_numbers = list(results_by_heuristic.keys())
    heuristics = list(results_by_heuristic[folder_numbers[0]].keys())
    heuristic_labels = [cfg.labels[c] for c in heuristics]

    for f, a in zip(folder_numbers, ax):
        np.random.seed(0)  # to control placement of strip plot data points
        sns.boxplot(
            data=pd.DataFrame(results_by_heuristic[f]),
            ax=a,
            palette="light:cornflowerblue",
            linewidth=1,
            fliersize=3,
        )
        sns.stripplot(
            data=pd.DataFrame(results_by_heuristic[f]),
            ax=a,
            palette="dark:darkgray",
            size=3,
        )
        a.set(title=f"c={coop} - {f}")
    for a in ax.flatten():
        a.xaxis.set_ticks(np.arange(len(heuristic_labels)))
        a.set_xticklabels(
            heuristic_labels,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        a.tick_params(axis="both", which="minor", labelsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Information-theoretic analysis of LBF experiments."
    )
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v7",
        type=str,
        help="Path to experimental results",
    )
    args = parser.parse_args()

    main(args.path)
