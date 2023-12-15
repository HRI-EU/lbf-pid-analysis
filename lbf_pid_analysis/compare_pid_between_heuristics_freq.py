#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Perform frequentist t-tests between PID estimates for different heuristics.
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

from mi_estimation import JidtDiscreteH
from utils import initialize_logger, read_settings
import config as cfg


def main(path, settings_path, folders, render):
    """Run comparison.

    Parameters
    ----------
    path : str
        Path to load PID estimates from
    settings_path : str
        Path to load analysis settings from
    folders : iterable
        Folder numbers over which to run comparison
    render : bool
        Whether to display generated figures
    """
    initialize_logger(
        log_name=f"compare_heuristics_{folders[0]}_to_{folders[-1]}"
    )
    JidtDiscreteH()  # needed to start the JVM
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    settings = read_settings(Path(settings_path))

    compare_pid_between_heuristics(
        folders,
        loadpath,
        savename=loadpath.joinpath(
            f"compare_heuristics_{folders[0]}_to_{folders[-1]}.csv"
        ),
        settings=settings,
        render=render,
    )


def compare_pid_between_heuristics(
    folders, loadpath, savename, settings, render=False
):
    """Perform statistical test to compare estimates for one heuristic against the other.

    Available heuristics:
        "MH_BASELINE": "BL",
        "MH_EGOISTIC": "EGO",  # former H1
        "MH_SOCIAL1": "SOC1",  # former H2
        "MH_COOPERATIVE": "COOP",  # value function, coop
        "MH_ADAPTIVE": "ADAPT",  # value function, coop or egoistic

    Parameters
    ----------
    folders : iterable
        Folders over which to run comparison
    loadpath : pathlib.Path
        Path to load estimates from
    savename : pathlib.Path
        Save path
    settings : dict
        Analysis settings containing the critical alpha level ('alpha) and a
        random seed ('seed')
    render : bool, optional
        Whether to visualize results, by default False
    """
    stats = []
    for dv in cfg.statistical_tests["dependent_variables"]:
        for comparison in cfg.statistical_tests["comparisons"]:
            if type(dv) is list:
                measure = dv[0]
                target_variable = dv[1]
            else:
                measure = dv
                target_variable = "any_food"

            with open(
                loadpath.joinpath(
                    f"mi_estimates_over_trials_{folders[0]}_to_{folders[-1]}_t_{target_variable}.p"
                ),
                "rb",
            ) as f:
                results = pickle.load(f)

            logging.info(
                "Analyzing measure %s (target var: %s) - %s vs. %s",
                measure,
                target_variable,
                comparison[0],
                comparison[1],
            )

            x1 = np.array(results[measure][comparison[0]])
            x2 = np.array(results[measure][comparison[1]])
            if measure == "n_cooperative_actions":
                x1[np.isnan(x1)] = 0
                x2[np.isnan(x2)] = 0
            significance, p_value = permutation_test(
                x1,
                x2,
                alpha=settings["alpha"],
                seed=settings["seed"],
                render=render,
            )
            stats.append(
                {
                    "measure": f"{measure}_{target_variable}",
                    "group1": comparison[0],
                    "group2": comparison[1],
                    "sign": significance,
                    "sign_corrected": (
                        p_value
                        < settings["alpha"]
                        / len(cfg.statistical_tests["comparisons"])
                    ),
                    "p_value": p_value,
                }
            )
    stats = pd.DataFrame(stats)
    stats.to_csv(savename, index=False)
    print(stats)


def permutation_test(x1, x2, alpha, seed=1, render=False):
    """Perform permutation test between two sets of estimates.

    Parameters
    ----------
    x1 : numpy.ndarray
        Set 1
    x2 : numpy.ndarray
        Set 2
    alpha : float
        Critical alpha level
    seed : int, optional
        Random seed, by default 1
    render : bool, optional
        Whether to plot generated test distribution, by default False

    Returns
    -------
    bool
        Whether test was statistically significant
    float
        Estimated p-value
    """
    np.random.seed(seed)
    if np.isnan(x1).any() or np.isnan(x2).any():
        raise RuntimeError("Input variables contain NaNs")
    if x1.shape != x2.shape:
        raise RuntimeError(
            "Input variables for statistical testing have to have equal size"
        )
    if np.array_equal(x1, x2):
        logging.info("Arrays are equal, do not perform a permutation test")
        return False, 1.0

    # n_perm = settings['analysis'].get('nperm', 1000)
    n_perm = 1000  # required to make the corrected alpha level achievable
    logging.info(
        "Performing statistical tests with n_perm=%s and alpha=%s",
        n_perm,
        alpha,
    )

    surrogate_distribution = np.zeros(n_perm)
    stat = abs(np.mean(x2) - np.mean(x1))
    for p in range(n_perm):
        x_combined = np.hstack((x1.copy(), x2.copy()))
        mask_1 = np.random.choice(
            np.arange(len(x_combined)), len(x1), replace=False
        )
        mask_2 = np.array(
            list(set(np.arange(len(x_combined))).difference(set(mask_1)))
        )
        surrogate_distribution[p] = abs(
            np.mean(x_combined[mask_1]) - np.mean(x_combined[mask_2])
        )

    # Calculate p-value.
    n_extreme = (surrogate_distribution > stat).sum()  # one-sided test
    if n_extreme == 0:
        p_value = 1 / n_perm
    else:
        p_value = n_extreme / len(surrogate_distribution)
    significance = p_value < alpha

    if render:
        plt.figure()
        plt.hist(surrogate_distribution)
        plt.axvline(np.percentile(surrogate_distribution, (1 - alpha) * 100))
        plt.axvline(stat)
    logging.debug(  # pylint: disable=W1201
        "Surrogate distribution: %s"  # pylint: disable=C0209
        % surrogate_distribution
    )

    return significance, p_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare information-theoretic estimates of LBF experiments between heuristics."
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
        "-s",
        "--settings",
        default="../settings/analysis_settings.yml",
        type=str,
        help="Path to analysis settings file",
    )
    parser.add_argument(
        "-f",
        "--folders",
        default=[1, 2, 3, 4, 5],
        nargs="+",
        help="List of folder numbers to analyze (e.g., all heuristics for one value of c)",
    )
    args = parser.parse_args()

    main(args.path, args.settings, args.folders, args.render)
