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


def main(path, settings_path, folders, target_variable):
    initialize_logger(log_name=f"estimate_local_sx_pid_t_{target_variable}")
    outpath = generate_output_path(path)
    local_pid_outpath = Path(outpath).joinpath("local_sx_pid")
    local_pid_outpath.mkdir(parents=True, exist_ok=True)

    for folder_number in range(1, folders + 1):
        # Identify files collected for experiment. Load settings file.
        foldername = Path(path, f"{int(folder_number):02d}*")
        trial_files = get_trial_file_names(foldername)
        experiment_settings = read_settings(
            glob.glob(str(foldername.joinpath("experiment_settings.yml")))[0]
        )
        analysis_settings = read_settings(Path(settings_path))

        # Create data structures to collect data over trials.
        heuristic = get_heuristic_used(experiment_settings)
        for trial in range(experiment_settings["experiment"]["ntrials"]):
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
                experiment_data,
                experiment_settings,
                trial,
                outpath=None,
                render=False,
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
            df = calculate_weighted_local_pid(pid_sx["local"])
            df.to_csv(local_pid_outpath.joinpath(f"{filename}.csv"))


def calculate_weighted_local_pid(local_pid):
    local_pid["shd*p"] = local_pid[cfg.col_shd] * local_pid[cfg.col_joint_prob]
    local_pid["syn*p"] = local_pid[cfg.col_syn] * local_pid[cfg.col_joint_prob]
    local_pid["unq_s1*p"] = local_pid[cfg.col_unq_s1] * local_pid[cfg.col_joint_prob]
    local_pid["unq_s2*p"] = local_pid[cfg.col_unq_s2] * local_pid[cfg.col_joint_prob]
    local_pid["mi*p"] = local_pid[cfg.col_lmi_s1_s2_t] * local_pid[cfg.col_joint_prob]
    return local_pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate local SxPID for LBF experiments."
    )
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v13",
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
        "--folders",
        default=55,
        type=int,
        help="Number of the folders to analyze",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    args = parser.parse_args()

    main(args.path, args.settings, args.folders, args.target)
