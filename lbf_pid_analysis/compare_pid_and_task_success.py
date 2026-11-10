#!/usr/bin/env python
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
import argparse
import logging
from pathlib import Path

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_measure_by_c import load_experiment_results
from utils import initialize_logger

FILETYPE = "svg"  # "png"


def main(
    path: str,
    target_variable: str,
    estimator: str,
    measure_task_success: str,
    coop_plot: float,
    n_folders: int,
    render: bool,
) -> None:
    initialize_logger(log_name=f"compare_measure_by_c_t_{target_variable}")

    loadpath = Path(path).joinpath("results_single_trial_analysis")

    results, _ = load_experiment_results(
        loadpath,
        target_variable,
        max_folder=n_folders,
    )

    compare_pid_and_task_success(
        results,
        estimator,
        measure_task_success,
        coop_plot,
        filename=loadpath.joinpath(
            f"comparison_pid_{estimator}_"
            f"task_success_{measure_task_success}_"
            f"c_{str(coop_plot).replace('.', '_')}.{FILETYPE}"
        ),
        render=render,
    )


def compare_pid_and_task_success(
    results: dict,
    estimator: str,
    measure_task_success: str,
    coop_plot: float,
    filename: Path,
    heuristics_used: list[str] | None = None,
    render: bool = False,
) -> None:
    if heuristics_used is None:
        heuristics_used = [
            "MH_EGOISTIC",
            "MH_SOCIAL1",
            "MH_ADAPTIVE",
            "MH_COOPERATIVE",
        ]

    try:
        results[coop_plot]
    except KeyError as exc:
        raise KeyError(
            f"Unknown cooperation parameter {coop_plot}, available parameters: {list(results.keys())}"
        ) from exc

    try:
        results[coop_plot][f"syn_{estimator}"]
    except KeyError as exc:
        raise KeyError(
            f"Unknown measure/estimator {estimator}, available measures: {list(results[coop_plot].keys())}"
        ) from exc

    pid_atoms = [
        f"{cfg.col_unq_s1}_{estimator}",
        f"{cfg.col_shd}_{estimator}",
        f"{cfg.col_syn}_{estimator}",
        f"{cfg.col_unq_s2}_{estimator}",
    ]

    box_colors = [
        cfg.colors["green_muted_lighter_1"],
        cfg.colors["red_muted_lighter_1"],
        cfg.colors["blue_muted_lighter_1"],
        cfg.colors["green_muted_lighter_1"],
    ]

    # Use an outer grid to separate task success and total MI from the
    # compact block of four PID panels.
    fig = plt.figure(
        figsize=(
            cfg.plot_elements["max_width_in"],
            cfg.plot_elements["row_height"] * 0.7,
        )
    )

    outer_grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 1.0, 4.0],
        wspace=0.45,
    )

    ax = [
        fig.add_subplot(outer_grid[0, 0]),
        fig.add_subplot(outer_grid[0, 1]),
    ]

    # The PID panels share one y-axis and are positioned close together.
    pid_grid = outer_grid[0, 2].subgridspec(
        nrows=1,
        ncols=len(pid_atoms),
        wspace=0.08,
    )

    pid_axes = []
    for i in range(len(pid_atoms)):
        if i == 0:
            pid_axes.append(fig.add_subplot(pid_grid[0, i]))
        else:
            pid_axes.append(
                fig.add_subplot(
                    pid_grid[0, i],
                    sharey=pid_axes[0],
                )
            )

    ax.extend(pid_axes)

    # Select the total mutual-information measure corresponding to the
    # requested estimator.
    if estimator.split("_")[0] == "norm":
        mi_measure = f"mi_{estimator[5:]}"
    else:
        mi_measure = f"mi_{estimator}"

    # Plot task success and total mutual information.
    for measure, current_ax in zip(
        [measure_task_success, mi_measure],
        ax[:2],
    ):
        try:
            df = pd.DataFrame(results[coop_plot][measure])[heuristics_used]
        except KeyError as exc:
            raise KeyError(
                f"Unknown measure {measure}, available measures: {list(results[coop_plot].keys())}"
            ) from exc

        df_long = df.melt(
            var_name="heuristic",
            value_name=measure,
        )
        sns.boxplot(
            x="heuristic",
            y=measure,
            data=df_long,
            order=heuristics_used,
            ax=current_ax,
            color=cfg.colors["gray"],
            width=0.6,
        )

        current_ax.set_xlabel("")
        current_ax.set_ylabel(
            cfg.labels["jmi"] if measure[:2] == "mi" else cfg.labels[measure]
        )
        current_ax.set_xticks(np.arange(len(heuristics_used)))
        current_ax.set_xticklabels(
            [cfg.labels[h] for h in heuristics_used],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

    # Plot one PID atom per panel and compare all policies within the panel.
    for i, (pid_atom, box_color) in enumerate(zip(pid_atoms, box_colors)):
        try:
            df = pd.DataFrame(results[coop_plot][pid_atom])[heuristics_used]
        except KeyError as exc:
            raise KeyError(
                f"Unknown PID atom {pid_atom}, available measures: {list(results[coop_plot].keys())}"
            ) from exc

        df_long = df.melt(
            var_name="heuristic",
            value_name=pid_atom,
        )

        pid_ax = ax[i + 2]

        print(pid_atom)
        sns.boxplot(
            x="heuristic",
            y=pid_atom,
            data=df_long,
            order=heuristics_used,
            ax=pid_ax,
            color=box_color,
            width=0.6,
        )

        pid_ax.set_xlabel("")
        pid_ax.set_ylabel(cfg.labels[estimator] if i == 0 else "")
        pid_ax.set_xticks(np.arange(len(heuristics_used)))
        pid_ax.set_xticklabels(
            [cfg.labels[h] for h in heuristics_used],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

    # Show y-axis ticks and labels only on the first PID panel.
    for pid_ax in ax[3:]:
        pid_ax.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

    plt.tight_layout()

    logging.info("Saving figure to %s", filename)
    plt.savefig(filename)

    if render:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Contrast estimated PID profiles and task success for "
            "different agent policies."
        )
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiment_results",
        type=str,
        help="Path to experimental results",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help="Variable to use as target in PID estimation",
    )
    parser.add_argument(
        "-e",
        "--estimator",
        type=str,
        default="norm_sx_cor",
        help="Estimator results to plot",
    )
    parser.add_argument(
        "-s",
        "--success",
        default="total_food_value_collected",
        type=str,
        help="Variable to use as measure of task success",
    )
    parser.add_argument(
        "-c",
        default=0.8,
        type=float,
        help="Cooperation parameter for which to compare",
    )
    parser.add_argument(
        "-f",
        "--folders",
        default=25,
        type=int,
        help="Number of results folders to parse",
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
