#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contrast synergy and task success for asymmetric agent strenghts, to show
# that both measures of agent behavior can diverge.
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


def main(path, render):
    initialize_logger(log_name="compare_synergy_asymmetric")
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    results_0 = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_0",
        folder_steps=2,
        max_folder=10,
    )
    results_1 = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_1",
        folder_steps=2,
        max_folder=10,
    )
    compare_syn_asymmetric(
        results_0,
        results_1,
        filename=loadpath.joinpath("compare_syn_asymmetric.pdf"),
        render=render,
    )


def compare_syn_asymmetric(
    results_0,
    results_1,
    filename,
    heuristics_used=None,
    coop_values=None,
    render=False,
):
    if coop_values is None:
        coop_values = COOP_PARAMS
    measure_syn = "syn_norm_sx_cor"  # 'syn', 'syn_norm'
    measure_unq_own_target = "unq_own_norm_sx_cor"  # 'syn', 'syn_norm'
    measure_unq_other_target = "unq_other_norm_sx_cor"  # 'syn', 'syn_norm'
    if heuristics_used is None:
        heuristics_used = list(results_0[0.0][measure_syn].keys())
    set_layout()
    try:
        results_0[0.0][measure_syn]
    except KeyError as e:
        logging.info(
            "Unknown measure %s, available measures: %s",
            measure_syn,
            list(results_0[0.0].keys()),
        )
        raise e

    for c in results_0.keys():
        results_0[c][measure_unq_own_target] = {}
        results_0[c][measure_unq_other_target] = {}
        results_1[c][measure_unq_own_target] = {}
        results_1[c][measure_unq_other_target] = {}
        for h in results_0[c][measure_syn].keys():
            mi_0 = np.array(results_0[c]["mi_sx_cor"][h], dtype=float)
            results_0[c][measure_unq_own_target][h] = np.divide(
                np.array(results_0[c]["unq1_sx_cor"][h]),
                mi_0,
                out=np.zeros_like(
                    np.array(results_0[c]["unq1_sx_cor"][h]), dtype=float
                ),
                where=mi_0 != 0,
            )
            results_0[c][measure_unq_other_target][h] = np.divide(
                np.array(results_0[c]["unq2_sx_cor"][h]),
                mi_0,
                out=np.zeros_like(
                    np.array(results_0[c]["unq2_sx_cor"][h]), dtype=float
                ),
                where=mi_0 != 0,
            )

            mi_1 = np.array(results_1[c]["mi_sx_cor"][h], dtype=float)
            results_1[c][measure_unq_own_target][h] = np.divide(
                np.array(results_1[c]["unq2_sx_cor"][h]),
                mi_1,
                out=np.zeros_like(
                    np.array(results_1[c]["unq2_sx_cor"][h]), dtype=float
                ),
                where=mi_1 != 0,
            )
            results_1[c][measure_unq_other_target][h] = np.divide(
                np.array(results_1[c]["unq1_sx_cor"][h]),
                mi_1,
                out=np.zeros_like(
                    np.array(results_1[c]["unq1_sx_cor"][h]), dtype=float
                ),
                where=mi_1 != 0,
            )

            if not np.array(results_0[c]["mi_sx_cor"][h]).all():
                print("Zero MI in heuristic", h)
            if not np.array(results_1[c]["mi_sx_cor"][h]).all():
                print("Zero MI in heuristic", h)

    fig, ax = plt.subplots(
        ncols=len(coop_values),
        nrows=3,
        figsize=(
            len(coop_values)
            * cfg.plot_elements["figsize"][  # pylint: disable=no-member
                "colwidth_in"
            ]
            * 1.2,
            3
            * cfg.plot_elements["figsize"][  # pylint: disable=no-member
                "lineheight_in"
            ],
        ),
    )
    labels = [
        ["$I_{syn}(F_0)$", "$I_{syn}(F_1)$"],
        ["$I_{unq}(F_0;A_0)$", "$I_{unq}(F_1;A_1)$"],
        ["$I_{unq}(F_0;A_1)$", "$I_{unq}(F_1;A_0)$"],
    ]
    for a, measure, l in zip(
        ax,
        [measure_syn, measure_unq_own_target, measure_unq_other_target],
        labels,
    ):
        for i, coop in enumerate(coop_values):
            df1 = pd.DataFrame(results_0[coop][measure])[heuristics_used].melt(
                var_name="heuristic", value_name=f"{measure}_0"
            )
            df2 = pd.DataFrame(results_1[coop][measure])[heuristics_used].melt(
                var_name="heuristic", value_name=f"{measure}_1"
            )
            df1["measure"] = f"{measure}_0"
            df2["measure"] = f"{measure}_1"
            df1.rename(columns={f"{measure}_0": "value"}, inplace=True)
            df2.rename(columns={f"{measure}_1": "value"}, inplace=True)
            # df1['value'] = (df1['value']-df1['value'].mean())/df1['value'].std()
            # df2['value'] = (df2['value']-df2['value'].mean())/df2['value'].std()

            sns.boxplot(
                x="heuristic",
                y="value",
                data=pd.concat([df1, df2]),
                ax=a[i],
                color=cfg.colors["gray"],
                hue="measure",
                width=0.6,
                linewidth=cfg.plot_elements["box"]["linewidth"],
                fliersize=cfg.plot_elements["marker_size"],
            )
            a[i].get_legend().remove()
            a[i].set(xlabel="", ylabel="")
            # sns.stripplot(
            #     data=pd.DataFrame(results[coop][m]),
            #     ax=ax[j, i],
            #     color=cfg.colors['bluegray_4'],
            #     size=cfg.plot_elements["marker_size"],
            # )
            # if Y_LIM is not None:
            #     ax.set(ylim=yl)

            a[i].set(title=f"c={coop}")
        a[0].set(ylabel=cfg.labels[measure])
        a[-1].legend(ncol=1, labels=l)

    for a in ax.flatten():
        a.tick_params(direction="out", length=2)

    # ylim = [np.inf, -np.inf]
    # for a in ax:
    #     ylim[0] = np.min([a.get_ylim()[0], ylim[0]])
    #     ylim[1] = np.max([a.get_ylim()[1], ylim[1]])
    # plt.setp(ax, ylim=ylim)

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

    plt.tight_layout()
    # fig.subplots_adjust(left=0.24, bottom=0.4, wspace=0.1, right=0.98)
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
    args = parser.parse_args()

    main(args.path, args.render)
