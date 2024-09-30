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
import argparse
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

from utils import initialize_logger
from plot_measure_by_c import load_experiment_results
from analyze_pid_per_trial import generate_output_path
import config as cfg


def main(path, target_variable, n_folders, coop_params):
    initialize_logger(log_name=f"export_avg_local_sx_pid_t_{target_variable}")
    outpath = generate_output_path(path)
    local_pid_outpath = Path(outpath).joinpath("local_sx_pid_plots")
    local_pid_outpath.mkdir(parents=True, exist_ok=True)

    coop_params = [float(c) for c in coop_params]

    # Print local stats.
    results, _ = load_experiment_results(
        outpath, target_variable, max_folder=n_folders, load_local_pid=True
    )
    export_avg_local_pid(
        results,
        coop_params,
        local_pid_outpath,
        filename=f"avg_local_sx_pid_t_{target_variable}",
    )


def export_avg_local_pid(
    results, coop_params, outpath, filename, weighted=True, write_tex=True
):
    """Export average local SxPID estimates as text files

    Export average local SxPID estimates with their standard deviation as csv
    files and optionally as additional tex tables.

    Parameters
    ----------
    results : dict
        SxPID estimates, including local SxPID results
    coop_params : iterable
        List of cooperation parameters for which to export estimates
    outpath : pathlib.Path
        Save path
    filename : str
        Save name
    weighted : bool, optional
        Whether to export local SxPID estimates weighted by their probability,
        by default True
    write_tex : bool, optional
        Whether to additionally export a tex table, by default True
    """
    heuristics = results[coop_params[0]]["local_sx_pid"].keys()

    columns = ["p(s1,s2,t)"]
    if weighted:
        columns += ["mi*p", "unq_s1*p", "unq_s2*p", "shd*p", "syn*p"]
    else:
        columns += ["i(s1,s2;t)", "unq_s1", "unq_s2", "shd", "syn"]
    # Define order of columns in exported csv/tex files.
    # export_columns = [columns[0]] + [c + " m" for c in columns[1:]] + [c + " sd" for c in columns[1:]]
    export_columns = [columns[0]]
    for c in columns[1:]:
        export_columns.append(c + " m")
        export_columns.append(c + " sd")

    for heuristic in heuristics:
        avg_lpid_heuristic = {}
        for coop_param in coop_params:
            local_pid = results[coop_param]["local_sx_pid"][heuristic]
            for df in local_pid:
                df.sort_values(
                    by=[cfg.col_t, cfg.col_s1, cfg.col_s2],
                    ascending=False,
                    inplace=True,
                )
                df["realizations"] = [
                    tuple(r)
                    for r in np.array(
                        [df["s1"].values, df["s2"].values, df["t"].values]
                    ).T
                ]
                df.set_index("realizations", inplace=True)
                df.drop(columns=["s1", "s2", "t"], inplace=True)
            df = pd.concat(local_pid, keys=np.arange(len(local_pid)))
            df_counts = pd.DataFrame(
                [df.xs(col, axis=1).unstack().count() for col in df.columns],
                index=df.columns,
            ).T
            df_means = pd.DataFrame(
                [df.xs(col, axis=1).unstack().mean() for col in df.columns],
                index=df.columns,
            ).T
            df_sds = pd.DataFrame(
                [df.xs(col, axis=1).unstack().std(ddof=1) for col in df.columns],
                index=df.columns,
            ).T.fillna(0)

            logging.info(
                "Statistics for local PID estimates over all trials: %s, c=%.2f\nMeans:\n%s\nSD:\n%s\nCounts\n%s",
                heuristic,
                coop_param,
                df_means[columns],
                df_sds[columns],
                df_counts[columns],
            )
            df_stats = df_means.join(df_sds[columns[1:]], lsuffix=" m", rsuffix=" sd")
            df_stats = df_stats[
                [columns[0]]
                + [f"{c} m" for c in columns[1:]]
                + [f"{c} sd" for c in columns[1:]]
            ]
            df_stats = df_stats[export_columns]
            df_stats.index.name = None
            avg_lpid_heuristic[coop_param] = df_stats
        avg_lpid_heuristic = pd.concat(
            avg_lpid_heuristic.values(),
            axis=1,
            keys=list(avg_lpid_heuristic.keys()),
            names=["c", "measure"],
        )
        avg_lpid_heuristic.fillna(0, inplace=True)
        avg_lpid_heuristic.to_csv(
            outpath.joinpath(f"{filename}_{heuristic}.csv"), float_format="%.4f"
        )
        if write_tex:
            with open(outpath.joinpath(f"{filename}_{heuristic}.tex"), "w") as tf:
                tf.write(
                    avg_lpid_heuristic.to_latex(
                        index=True,
                        formatters={"name": str.upper},
                        float_format="{:.4f}".format,
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export local average SxPID estimates."
    )
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
    parser.add_argument(
        "-c",
        "--coop_params",
        nargs="+",
        default=np.arange(0.0, 1.1, 0.1),
        help=("Cooperation parameters to export"),
    )
    args = parser.parse_args()

    main(
        args.path,
        args.target,
        args.total_folders,
        args.coop_params,
    )
