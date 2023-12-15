#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Config file -- specifies variables that are used over modules.
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
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

from utils import initialize_logger

FOLDERS = [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25]]
COOP_PARAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(path, stats_type, measure):
    """Run collection of statistical testing results and write to log file"""
    initialize_logger(log_name=f"summary_group_comparisons_{measure}")
    summarize_statistical_tests(path, stats_type, measure)


def _pivot_results(stats, column, dependent_variable="all"):
    """Convert and plot statistical results for readability"""
    if dependent_variable == "all":
        dependent_variables = stats["measure"].unique()
    else:
        dependent_variables = [dependent_variable]

    for dv in dependent_variables:
        print("\n")
        df_m = stats[stats["measure"] == dv]
        logging.info(  # pylint: disable=W1201
            "--- %s group2 > group1\n%s"  # pylint: disable=C0209
            % (
                dv,
                df_m.pivot_table(column, ["group1"], "group2").reindex(
                    ["MH_SOCIAL1", "MH_COOPERATIVE", "MH_ADAPTIVE"], axis=1
                ),
            )
        )


def _initialize_summary_file(loadpath, measure):
    """Create a sumary file for Bayesian results"""
    summary_file = loadpath.joinpath(f"bayes_results_summary_{measure}.csv")
    summary_infos = [
        "group1",
        "group2",
        "effect_size_mean",
        "effect_size_sd",
        "effect_size_hdi3%",
        "effect_size_hdi97%",
        "es_outside_rope",
    ]
    print({"": h for h in summary_infos})
    pd.DataFrame({h: h for h in summary_infos}, index=[0]).to_csv(
        summary_file, header=False, index=False
    )
    return summary_file, summary_infos


def _add_to_summary(df, title, summary_file):
    """Add a batch of Bayesian results to summary file"""
    title = pd.DataFrame({"": title}, index=[0])
    title.to_csv(summary_file, mode="a", index=False, header=False)
    df.to_csv(summary_file, mode="a", sep="&", index=False, header=False)


def summarize_statistical_tests(path, stats_type="both", measure="all"):
    """Summarize results of statistical tests in specified path

    Parameters
    ----------
    path : str
        Path to results
    stats_type : str, optional
        Statistics to use ('freq', 'bayes', or 'both'), by default 'both'
    measure : str, optional
        Measure to use as dependent variable, by default 'all'
    """
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    summary_file, summary_infos = _initialize_summary_file(loadpath, measure)

    for f, folder in enumerate(FOLDERS):
        logging.info(  # pylint: disable=W1201
            "Statistical tests, c=%4.2f (folders %d to %d)"  # pylint: disable=C0209
            % (COOP_PARAMS[f], folder[0], folder[-1])
        )
        try:
            stats_bayes = pd.read_csv(
                loadpath.joinpath(
                    f"compare_heuristics_bayes_{folder[0]}_to_{folder[-1]}.csv"
                )
            )
        except FileNotFoundError as e:
            logging.info(e)
            continue
        stats_freq = pd.read_csv(
            loadpath.joinpath(
                f"compare_heuristics_{folder[0]}_to_{folder[-1]}.csv"
            )
        )
        _add_to_summary(
            stats_bayes[summary_infos].loc[stats_bayes["measure"] == measure],
            title=f"c={COOP_PARAMS[f]:4.2f}",
            summary_file=summary_file,
        )
        if stats_type in ["bayes", "both"]:
            logging.info("Bayesian results")
            _pivot_results(
                stats_bayes, column="mean_nonzero", dependent_variable=measure
            )

        if stats_type in ["freq", "both"]:
            logging.info("Frequentist results (corrected)")
            _pivot_results(
                stats_freq, column="sign_corrected", dependent_variable=measure
            )
        if stats_type == "both":
            logging.info("Congruent results")
            stats_congruent = pd.DataFrame(
                {
                    "group1": stats_bayes["group1"],
                    "group2": stats_bayes["group2"],
                    "measure": stats_bayes["measure"],
                    "congruent": stats_bayes["mean_nonzero"]
                    == stats_freq["sign_corrected"],
                }
            )
            _pivot_results(
                stats_congruent, column="congruent", dependent_variable=measure
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Information-theoretic analysis of LBF experiments."
    )
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/",
        type=str,
        help="Path to experimental results",
    )
    parser.add_argument(
        "-s",
        "--stats_type",
        default="bayes",
        type=str,
        help="Statistics to use ('freq', 'bayes', or 'both')",
    )
    parser.add_argument(
        "-m",
        "--measure",
        default="syn_norm_sx_cor_any_food",
        type=str,
        help="Measure to report ('syn_norm_sx_cor_any_food', etc.)",
    )
    args = parser.parse_args()

    main(args.path, args.stats_type, args.measure)
