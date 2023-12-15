#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generate results figures used in the paper.
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
import pathlib
from pathlib import Path
import logging
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from utils import initialize_logger
from plot_measure_by_c import load_experiment_results
import config as cfg

MAX_FOLDER = 30
FOLDER_STEPS = 6

SYN_MEASURE = "syn_norm_sx_cor"
PLOT_STRIPPLOT = False
SIGNIFICANT_ONLY = False  # plot only significant trials
PLOT_BOX_PLOT_OUTLIERS = False
LINEPLOT = False


def set_paper_layout():
    """Set rcParams for matplotlib and define additional colors for figures

    Settings generate figures that can be included in manuscript with minor
    changes. Linewidths etc. conform to journal guidelines.
    """
    cfg.colors["bl"] = cfg.colors["yellow"]
    cfg.colors["ego"] = cfg.colors["purple"]
    cfg.colors["social"] = cfg.colors["green"]
    cfg.colors["coop"] = cfg.colors["red"]
    cfg.colors["adapt"] = cfg.colors["blue"]
    cfg.colors["adapt_light"] = cfg.colors["blue_light_2"]

    plt.rc("text", usetex=False)
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    # plt.rc(
    #     "font",
    #     **{"family": cfg.plot_elements["fontfamily"], "serif": ["Times"]},
    # )
    # plt.rc('text', usetex=True)
    plt.rc(
        "font", size=cfg.plot_elements["textsize"]["large"]
    )  # controls default text sizes
    plt.rc(
        "axes", titlesize=cfg.plot_elements["textsize"]["large"]
    )  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=cfg.plot_elements["textsize"]["large"]
    )  # fontsize of the x and y labels
    plt.rc(
        "xtick", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the tick labels
    plt.rc(
        "ytick", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the tick labels
    plt.rc(
        "legend", fontsize=cfg.plot_elements["textsize"]["medium"]
    )  # legend fontsize
    plt.rc(
        "figure", titlesize=cfg.plot_elements["textsize"]["large"]
    )  # fontsize of the figure title


def load_selected_results(loadpath, coop, heuristics, target_variable, measure):
    """Load estimates from disk"""
    results = {}
    for c in coop:
        searchstring = str(loadpath.joinpath(f"*_s25_coop{c}_MHTrue_*"))
        results_folders = glob.glob(searchstring)
        if len(results_folders) == 0:
            raise FileNotFoundError(
                "Did not find any file for string", searchstring
            )
        if type(loadpath) is pathlib.PosixPath:
            folder_numbers = np.sort(
                [name.split("/")[-1][:2] for name in results_folders]
            )
        else:
            folder_numbers = np.sort(
                [name.split("\\")[-1][:2] for name in results_folders]
            )
        with open(
            loadpath.joinpath(
                "results_single_trial_analysis",
                f"mi_estimates_over_trials_{int(folder_numbers[0])}_to_{int(folder_numbers[-1])}_t_{target_variable}.p",
            ),
            "rb",
        ) as f:
            estimates = pickle.load(f)
        results[c] = {}
        for h in heuristics:
            for pid_atom in [
                "syn_sx_cor",
                "shd_sx_cor",
                "unq1_sx_cor",
                "unq2_sx_cor",
            ]:
                if np.sum(np.array(estimates[pid_atom][h]) < 0) > 0:
                    print(pid_atom, estimates[pid_atom][h])
                    print("Negative PID atom")
            mask = np.invert(estimates["jmi_sign"][h])
            results[c][h] = np.array(estimates[measure][h], dtype=float)
            if SIGNIFICANT_ONLY:
                results[c][h][mask] = np.nan
                if mask.any():
                    print(f"{np.sum(mask)} non-sign. trials for {h}, c={c}")
                else:
                    print(f"All trials significant for {h}, c={c}")

    return results


def mask_non_significant(results):
    for c in list(results.keys()):
        for m in list(results[c].keys()):
            if m == "jmi_sign":
                continue
            for h in list(results[c][m].keys()):
                mask = np.invert(results[c]["jmi_sign"][h])
                results[c][m][h] = np.array(results[c][m][h], dtype=float)
                results[c][m][h][mask] = np.nan
                if mask.any():
                    print(f"{np.sum(mask)} non-sign. trials for {h}, c={c}")
    return results


def _add_grid(ax, axis="y"):
    ax.grid(
        color=cfg.colors["gray_light_2"],
        axis=axis,
        which="major",
        lw=0.5,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _set_axis_style(ax):
    ax.spines["bottom"].set_color(cfg.colors["gray_light_2"])
    ax.spines["top"].set_color(cfg.colors["gray_light_2"])
    ax.spines["right"].set_color(cfg.colors["gray_light_2"])
    ax.spines["left"].set_color(cfg.colors["gray_light_2"])


def _set_heuristic_xtick_labels(ax, heuristics, rotate=35):
    heuristic_labels = [cfg.labels[h] for h in heuristics]
    # sns.despine(offset=2, trim=True)
    for a in ax.flatten():
        a.tick_params(length=0)
        a.xaxis.set_ticks(
            np.arange(len(heuristic_labels))
        )  # to avoid FixedFormatter warning
        a.set_xticklabels(
            heuristic_labels,
            rotation=rotate,
            ha="right",
            rotation_mode="anchor",
        )


def _unify_ylim(ax):
    """Unify subplots' y-axis limits"""
    ylim = [np.inf, -np.inf]
    for a in ax:
        ylim[0] = np.min([a.get_ylim()[0], ylim[0]])
        ylim[1] = np.max([a.get_ylim()[1], ylim[1]])
    plt.setp(ax, ylim=ylim)


def plot_figure_asymmetric_results(resultspaths, savepath):
    set_paper_layout()

    n_figure_cols = 6
    n_figure_rows = 4
    savename = savepath.joinpath("asymmetric_condition.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            n_figure_cols * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )

    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    d = 0.0
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = "n_collections_agent_0"
    plot_synergy_norm_by_c(
        resultspaths, ax=ax[0, 1:], d=d, c=c, heuristics=h, t=t
    )
    h = ["MH_ADAPTIVE"]
    plot_asymmetric_synergy(resultspaths, ax=ax[1:, 1:], heuristics=h, c=c)

    # for i in range(len(ax)):
    #    ax[i, 0].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(left=0.15, wspace=0.2, right=0.99)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_figure_manipulation_check(resultspaths, savepath):
    """Check whether manipulation of environment and agent behavior worked

    Plot performance in terms of joint actions and task success to verify
    that performance matches expectation based on environment settings and
    agent abilities.
    """
    set_paper_layout()

    n_figure_cols = 5
    n_figure_rows = 4
    savename = savepath.joinpath("environment_manipulation_check.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            n_figure_cols * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )
    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    c_no_dist = [0.0, 0.25, 0.5, 0.75, 1.0]
    d = 0.5  # a higher distractor makes for a more clear result in the manipulation check
    c_dist = [0.0, 0.1, 0.2, 0.3, 0.4]
    # d = 0.2
    # c_dist = [0.0, 0.25, 0.5, 0.75, 0.8]
    measure = "n_cooperative_actions"
    plot_performance_comparison_dist_no_dist(
        resultspaths,
        ax=ax[:2, :],
        no_dist=0.0,
        dist=d,
        coop_no_dist=c_no_dist,
        coop_dist=c_dist,
        heuristics=h,
        measure=measure,
    )
    measure = "total_food_value_collected"
    plot_performance_comparison_dist_no_dist(
        resultspaths,
        ax=ax[2:, :],
        no_dist=0.0,
        dist=d,
        coop_no_dist=c_no_dist,
        coop_dist=c_dist,
        heuristics=h,
        measure=measure,
    )

    plt.tight_layout()
    # fig.subplots_adjust(wspace=0.1)
    logging.info("Saving figure to %s", savename)
    fig.savefig(savename)
    plt.show()


def plot_figure_syngergy_by_c_and_heuristic(resultspaths, savepath):
    """Plot first paper figures that shows synergy for heuristics

    Parameters
    ----------
    savepath : pathlib.Path
        Figure save path
    """
    set_paper_layout()

    n_figure_cols = 6
    n_figure_rows = 3
    savename = savepath.joinpath("synergy_by_c_and_heuristic.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            n_figure_cols * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )
    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    d = 0.0
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = "any_food"
    print("Plot results for non-distractor condition")
    if LINEPLOT:
        gs = ax[0, 0].get_gridspec()
        # remove the underlying axes
        for a in ax[0, :5]:
            a.remove()
        axbig = fig.add_subplot(gs[0, :5])
        h = [
            "MH_BASELINE",
            "MH_EGOISTIC",
            "MH_SOCIAL1",
            "MH_COOPERATIVE",
            "MH_ADAPTIVE",
        ]
        d = 0.0
        c = [0.0, 0.25, 0.5, 0.75, 1.0]
        t = "any_food"
        print("Plot results for non-distractor condition")
        plot_synergy_norm_by_c_lineplot(
            resultspaths, axbig, d, c, target_variable=t
        )
    else:
        plot_synergy_norm_by_c(resultspaths, ax[0, :5], d, c, heuristics=h, t=t)
    # h = ['MH_BASELINE', 'MH_EGOISTIC', 'MH_SOCIAL1', 'MH_COOPERATIVE', 'MH_ADAPTIVE']
    # d = 0.5
    # c = [0.1]
    h = ["MH_EGOISTIC", "MH_SOCIAL1", "MH_COOPERATIVE"]
    d = 0.2
    c = [0.8]
    t = "any_food"
    print(f"Plot results for distractor condition (d={d})")
    plot_synergy_norm_by_c(resultspaths, ax[0, 5], d, c, heuristics=h, t=t)

    d = 0.0
    # c = [0.25]
    c = [0.5]
    h = ["MH_EGOISTIC", "MH_ADAPTIVE", "MH_COOPERATIVE"]
    t = "any_food"
    print("Plot results for synergy vs. task success")
    plot_synergy_vs_task_success(
        resultspaths, ax[1, 5], d=d, c=c, heuristics=h, t=t
    )

    # Plot correlation between synergy and joint actions.
    t = "any_food"
    h = ["MH_EGOISTIC", "MH_COOPERATIVE"]
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    plot_correlation_syn_coop_actions(
        resultspaths, ax=ax[1:, :5], d=d, c=c, heuristics=h, t=t
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, right=0.9)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_synergy_norm_by_c(
    resultspaths, ax, d, c, heuristics, t, measure=SYN_MEASURE
):
    results = load_selected_results(
        loadpath=Path(resultspaths[d]).resolve(),
        coop=c,
        heuristics=heuristics,
        target_variable=t,
        measure=measure,
    )
    set_paper_layout()
    if type(ax) is not np.ndarray:
        ax = np.array([ax])
    box_color = [
        cfg.colors["bl"],
        cfg.colors["ego"],
        cfg.colors["social"],
        cfg.colors["coop"],
        cfg.colors["adapt"],
    ]
    sns.color_palette(box_color)
    matplotlib.rc("axes", edgecolor=cfg.colors["gray_light_2"])
    for i, coop in enumerate(c):
        sns.boxplot(
            data=pd.DataFrame(results[coop]),
            ax=ax[i],
            showfliers=PLOT_BOX_PLOT_OUTLIERS,
            palette=box_color,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
            zorder=3,
        )
        if PLOT_STRIPPLOT:
            sns.stripplot(
                data=pd.DataFrame(results[coop]),
                ax=ax[i],
                alpha=0.5,
                color=cfg.colors["gray"],
                size=cfg.plot_elements["marker_size"],
            )

        # ax[i].text(0.04, 0.88, f'c={coop}', fontweight='bold', transform=ax[i].transAxes)
        ax[i].set(title=f"$d={d}$\n$c={coop}$")
        _add_grid(ax[i])
        _set_axis_style(ax[i])
    ax[0].set(ylabel=cfg.labels[SYN_MEASURE])
    for a in ax[1:]:
        a.set(yticklabels=[])
    _unify_ylim(ax)

    _set_heuristic_xtick_labels(ax, heuristics)


def plot_synergy_vs_task_success(resultspaths, ax, d, c, heuristics, t):
    results = load_experiment_results(
        loadpath=Path(resultspaths[d]).joinpath(
            "results_single_trial_analysis"
        ),
        target_variable=t,
    )
    if SIGNIFICANT_ONLY:
        results = mask_non_significant(results)

    measure_task_success = "total_food_value_collected"  # n_collections -> no. collections performed, total value -> collected value
    set_paper_layout()
    try:
        results[0.0][SYN_MEASURE]
    except KeyError as e:
        logging.info(
            "Unknown measure %s, available measures: %s",
            SYN_MEASURE,
            list(results[0.0].keys()),
        )
        raise e

    if type(ax) is not np.ndarray:
        ax = np.array([ax])
    for i, coop in enumerate(c):
        try:
            df1 = pd.DataFrame(results[coop][measure_task_success])[
                heuristics
            ].melt(var_name="heuristic", value_name=measure_task_success)
        except KeyError as e:
            print(f"Available measures: {results[coop].keys()}")
            raise e
        df2 = pd.DataFrame(results[coop][SYN_MEASURE])[heuristics].melt(
            var_name="heuristic", value_name=SYN_MEASURE
        )
        df1["measure"] = measure_task_success
        df2["measure"] = SYN_MEASURE
        df1.rename(columns={measure_task_success: "value"}, inplace=True)
        df2.rename(columns={SYN_MEASURE: "value"}, inplace=True)
        df1["value"] = (df1["value"] - df1["value"].mean()) / df1["value"].std()
        df2["value"] = (df2["value"] - df2["value"].mean()) / df2["value"].std()

        sns.boxplot(
            x="heuristic",
            y="value",
            showfliers=PLOT_BOX_PLOT_OUTLIERS,
            data=pd.concat([df1, df2]),
            ax=ax[i],
            palette=[
                cfg.colors["blue"],
                cfg.colors["gray_light_2"],
            ],
            hue="measure",
            width=cfg.plot_elements["box"]["boxwidth"],
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        ax[i].get_legend().remove()
        ax[i].set(xlabel="", ylabel="", title=f"$d={d}$\n$c={coop}$")
        _add_grid(ax[i])
        _set_axis_style(ax[i])
        # sns.stripplot(
        #     data=pd.DataFrame(results[coop][m]),
        #     ax=ax[j, i],
        #     color=cfg.colors['bluegray_4'],
        #     size=cfg.plot_elements['markers']['size'],
        # )
        # if Y_LIM is not None:
        #     ax.set(ylim=yl)

    ax[0].set(ylabel="Z")
    for a in ax.flatten():
        a.tick_params(direction="out", length=2)
    ax[0].legend(loc="upper left", labels=["$F$", "$I_{syn}$"])
    _set_heuristic_xtick_labels(ax, heuristics)


def plot_correlation_syn_coop_actions(resultspaths, ax, d, c, t, heuristics):
    results = load_experiment_results(
        loadpath=Path(resultspaths[d]).joinpath(
            "results_single_trial_analysis"
        ),
        target_variable=t,
    )
    if SIGNIFICANT_ONLY:
        results = mask_non_significant(results)

    set_paper_layout()
    plt.rc(
        "ytick", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the tick labels
    plt.rc(
        "xtick", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the tick labels
    corr_measure = "syn_sx_cor"
    corr_coop_actions = "n_cooperative_actions"

    for i, heuristic in enumerate(heuristics):
        ylim = [np.inf, -np.inf]
        for j, coop in enumerate(c):
            try:
                x1 = np.array(results[coop][corr_coop_actions][heuristic])
            except KeyError as e:
                print(
                    f"Variables available for correlation: {results[coop].keys()}"
                )
                print(e.args)
                raise
            x2 = np.array(results[coop][corr_measure][heuristic])
            x1[np.isnan(x1)] = 0
            if len(np.unique(x1)) == 1 or len(np.unique(x2)) == 1:
                logging.info(
                    "No variation in input variable, skipping %s, c=%s",
                    heuristic,
                    coop,
                )
                continue

            # # Normalize variables
            # x1 = (x1-np.mean(x1))/np.std(x1)
            # x2 = (x2-np.mean(x2))/np.std(x2)
            # x1 = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
            # x2 = (x2-np.min(x2))/(np.max(x2)-np.min(x2))

            corr = np.corrcoef(x1, x2)[0, 1]
            # ax[i, j].scatter(
            #     x1, x2,
            #     c=cfg.colors['gray_dark'],
            #     s=cfg.plot_elements['scatter']['size'],
            #     marker=cfg.plot_elements['scatter']['marker'],
            #     alpha=cfg.plot_elements['scatter']['alpha'],
            #     edgecolors=cfg.plot_elements['scatter']['edgecolors'],
            #     linewidth=cfg.plot_elements['scatter']['linewidth'],
            # )
            sns.regplot(
                x=x1,
                y=x2,
                line_kws={
                    "color": "k",
                    "linewidth": cfg.plot_elements["linewidth"],
                },
                scatter_kws={
                    "color": cfg.colors["gray_dark"],
                    "s": cfg.plot_elements["scatter"]["size"],
                    "marker": cfg.plot_elements["scatter"]["marker"],
                    "alpha": cfg.plot_elements["scatter"]["alpha"],
                    "edgecolors": cfg.plot_elements["scatter"]["edgecolors"],
                    "linewidth": cfg.plot_elements["scatter"]["linewidth"],
                },
                ax=ax[i, j],
            )
            t = ax[i, j].text(
                0.1,
                0.08,
                f"$r={corr:.2f}$",
                fontweight="bold",
                transform=ax[i, j].transAxes,
            )
            t.set_bbox(dict(facecolor="w", alpha=0.6, lw=0))
            ax[0, j].set(title=f"$d={d}$\n$c={coop}$")
            ylim[0] = np.min([ax[i, j].get_ylim()[0], ylim[0]])
            ylim[1] = np.max([ax[i, j].get_ylim()[1], ylim[1]])
            _set_axis_style(ax[i, j])

        ax[i, 0].set(
            ylabel=f"{cfg.labels[heuristic]}\n{cfg.labels[corr_measure]}"
        )
        for a in ax[i, :]:
            a.set(ylim=ylim)
        for a in ax[i, 1:]:
            a.set(yticklabels=[])
        for a in ax.flatten():
            a.tick_params(
                direction="out",
                length=2,
                color=cfg.colors["gray_light_2"],
            )

    for a in ax[-1,]:
        a.set(xlabel=cfg.labels[corr_coop_actions])


def plot_performance_comparison_dist_no_dist(
    resultspaths,
    ax,
    no_dist,
    dist,
    coop_no_dist,
    coop_dist,
    measure,
    heuristics,
):
    results_no_dist = load_selected_results(
        loadpath=Path(resultspaths[no_dist]),
        coop=coop_no_dist,
        heuristics=heuristics,
        target_variable="any_food",
        measure=measure,
    )
    results_dist = load_selected_results(
        loadpath=Path(resultspaths[dist]),
        coop=coop_dist,
        heuristics=heuristics,
        target_variable="any_food",
        measure=measure,
    )
    set_paper_layout()

    plt.rc(
        "xtick", labelsize=cfg.plot_elements["textsize"]["medium"]
    )  # fontsize of the tick labels
    plt.rc("ytick", labelsize=cfg.plot_elements["textsize"]["medium"])
    for results, a, d in zip(
        [results_no_dist, results_dist], ax, [no_dist, dist]
    ):
        ylim = [np.inf, -np.inf]
        for i, heuristic in enumerate(heuristics):
            coop_by_c = {}
            for coop in results.keys():
                coop_by_c[coop] = results[coop][heuristic]
            sns.stripplot(
                data=pd.DataFrame(coop_by_c),
                ax=a[i],
                color=cfg.colors["gray_dark"],
                size=cfg.plot_elements["marker_size"],
            )
            a[i].errorbar(
                x=np.arange(len(coop_by_c)),
                y=pd.DataFrame(coop_by_c).mean(),
                yerr=pd.DataFrame(coop_by_c).std(),
                color=cfg.colors["gray"],
            )
            a[i].set(xlabel="$c$", title=f"{cfg.labels[heuristic]}\n$d={d}$")
            ylim[0] = np.min([ylim[0], a[i].get_ylim()[0]])
            ylim[1] = np.max([ylim[1], a[i].get_ylim()[1]])

        a[0].set(ylabel=cfg.labels[measure])

        if measure in ["n_cooperative_actions", "total_food_value_collected"]:
            for plot in a[2:]:
                plot.set(ylim=ylim)
        elif measure == "n_collections":
            for plot in a[1:]:
                plot.set(ylim=ylim)

    for a in ax.flatten():
        _set_axis_style(a)
        a.tick_params(
            direction="out", length=2, color=cfg.colors["gray_light_2"]
        )


def plot_asymmetric_synergy(resultspaths, ax, heuristics, c):
    loadpath = Path(resultspaths["asym"]).joinpath(
        "results_single_trial_analysis"
    )
    results_target_a1 = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_0",
        folder_steps=2,
        max_folder=10,
    )
    results_target_a2 = load_experiment_results(
        loadpath,
        target_variable="n_collections_agent_1",
        folder_steps=2,
        max_folder=10,
    )

    set_paper_layout()
    measure_syn = "syn_norm_sx_cor"
    measure_unq_own_target = "unq_own_norm_sx_cor"
    measure_unq_other_target = "unq_other_norm_sx_cor"

    try:
        results_target_a1[0.0][measure_syn]
    except KeyError as err:
        logging.info(
            "Unknown measure %s, available measures: %s",
            measure_syn,
            list(results_target_a1[0.0].keys()),
        )
        print(err.args)
        raise

    for coop in results_target_a1.keys():  # pylint: disable=C0206
        results_target_a1[coop][measure_unq_own_target] = {}
        results_target_a1[coop][measure_unq_other_target] = {}
        results_target_a2[coop][measure_unq_own_target] = {}
        results_target_a2[coop][measure_unq_other_target] = {}
        for h in results_target_a1[coop][measure_syn].keys():
            results_target_a1[coop][measure_unq_own_target][h] = np.array(
                results_target_a1[coop]["unq1_sx_cor"][h]
            ) / np.array(results_target_a1[coop]["mi_sx_cor"][h])
            results_target_a1[coop][measure_unq_other_target][h] = np.array(
                results_target_a1[coop]["unq2_sx_cor"][h]
            ) / np.array(results_target_a1[coop]["mi_sx_cor"][h])
            results_target_a2[coop][measure_unq_own_target][h] = np.array(
                results_target_a2[coop]["unq2_sx_cor"][h]
            ) / np.array(results_target_a2[coop]["mi_sx_cor"][h])
            results_target_a2[coop][measure_unq_other_target][h] = np.array(
                results_target_a2[coop]["unq1_sx_cor"][h]
            ) / np.array(results_target_a2[coop]["mi_sx_cor"][h])
            if (np.array(results_target_a1[coop]["mi_sx_cor"][h]) == 0).any():
                print("Zero entropy in some trials")
            if (np.array(results_target_a2[coop]["mi_sx_cor"][h]) == 0).any():
                print("Zero entropy in some trials")

    measures = [measure_syn, measure_unq_own_target, measure_unq_other_target]
    colors = [
        [cfg.colors["adapt"], cfg.colors["adapt_light"]],
        [cfg.colors["cyan"], cfg.colors["cyan_light_2"]],
        [cfg.colors["green"], cfg.colors["green_light_2"]],
    ]
    labels = [
        ["$I_{syn}(F_0)$", "$I_{syn}(F_1)$"],
        ["$I_{unq}(F_0;A_0)$", "$I_{unq}(F_1;A_1)$"],
        ["$I_{unq}(F_0;A_1)$", "$I_{unq}(F_1;A_0)$"],
    ]
    for a, measure, color, l in zip(ax, measures, colors, labels):
        ylim = [np.inf, -np.inf]
        for i, coop in enumerate(c):
            df1 = pd.DataFrame(results_target_a1[coop][measure])[
                heuristics
            ].melt(var_name="heuristic", value_name=f"{measure}_0")
            df2 = pd.DataFrame(results_target_a2[coop][measure])[
                heuristics
            ].melt(var_name="heuristic", value_name=f"{measure}_1")
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
                showfliers=PLOT_BOX_PLOT_OUTLIERS,
                ax=a[i],
                palette=color,
                hue="measure",
                width=0.6,
                linewidth=cfg.plot_elements["box"]["linewidth"],
                fliersize=cfg.plot_elements["marker_size"],
            )
            a[i].get_legend().remove()
            a[i].set(xlabel="", ylabel="", title=f"$c={coop}$")
            a[i].tick_params(
                axis="x",
                which="both",  # both major and minor
                bottom=False,
                labelbottom=False,
            )
            _add_grid(a[i])
            _set_axis_style(a[i])
            ylim[0] = np.min([ylim[0], a[i].get_ylim()[0]])
            ylim[1] = np.max([ylim[1], a[i].get_ylim()[1]])
            # sns.stripplot(
            #     data=pd.DataFrame(results[coop][m]),
            #     ax=ax[j, i],
            #     color=cfg.colors['bluegray_4'],
            #     size=cfg.plot_elements['markers']['size'],
            # )
            # if Y_LIM is not None:
            #     ax.set(ylim=yl)

        a[0].set(ylabel=cfg.labels[measure_syn])
        # a[-1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.35), fancybox=True, ncol=2, labels=['$F^{A_0}$', '$F^{A_1}$'])
        a[-1].legend(labels=l)
        # a[-1].legend()
        for plot in a:
            plot.set(ylim=ylim)
        for plot in a[1:]:
            plot.set(yticklabels=[])

    _set_heuristic_xtick_labels(ax, heuristics)
    for a in ax.flatten():
        a.tick_params(
            direction="out", length=2, color=cfg.colors["gray_light_2"]
        )


def plot_asymmetric_task_success(resultspaths, savepath):
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    d = 0.0
    heuristics = ["MH_ADAPTIVE"]
    loadpath = Path(resultspaths["asym"]).joinpath(
        "results_single_trial_analysis"
    )
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
    if SIGNIFICANT_ONLY:
        results_0 = mask_non_significant(results_0)
        results_1 = mask_non_significant(results_1)

    set_paper_layout()

    # measure_success_label = 'n_collections'
    # measure_success_0 = 'n_collections_agent_0'  # 'syn', 'syn_norm'
    # measure_success_1 = 'n_collections_agent_1'  # 'syn', 'syn_norm'
    measure_success_label = "total_food_value_collected"
    measure_success_0 = "food_value_collected_agent_0"  # 'syn', 'syn_norm'
    measure_success_1 = "food_value_collected_agent_1"  # 'syn', 'syn_norm'

    if (
        measure_success_0 not in results_0[0.0].keys()
        or measure_success_1 not in results_0[0.0].keys()
    ):
        raise KeyError(
            "Unknown measures %s or %s, available measures: %s"
            % (
                measure_success_0,
                measure_success_1,
                list(results_0[0.0].keys()),
            ),
        )

    fig, ax = plt.subplots(
        ncols=len(c),
        figsize=(
            len(c) * cfg.plot_elements["figsize"]["colwidth_in"] * 1.2,
            1.2 * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
        sharey=True,
    )
    for i, coop in enumerate(c):
        df1 = pd.DataFrame(results_0[coop][measure_success_0])[heuristics].melt(
            var_name="heuristic", value_name=measure_success_0
        )
        df2 = pd.DataFrame(results_1[coop][measure_success_1])[heuristics].melt(
            var_name="heuristic", value_name=measure_success_1
        )
        df1["measure"] = cfg.labels[measure_success_0]
        df2["measure"] = cfg.labels[measure_success_1]
        df1.rename(columns={measure_success_0: "value"}, inplace=True)
        df2.rename(columns={measure_success_1: "value"}, inplace=True)
        # df1['value'] = (df1['value']-df1['value'].mean())/df1['value'].std()
        # df2['value'] = (df2['value']-df2['value'].mean())/df2['value'].std()
        sns.boxplot(
            x="heuristic",
            y="value",
            data=pd.concat([df1, df2]),
            showfliers=PLOT_BOX_PLOT_OUTLIERS,
            ax=ax[i],
            palette=[
                cfg.colors["gray"],
                cfg.colors["gray_light_2"],
            ],
            hue="measure",
            width=0.6,
            linewidth=cfg.plot_elements["box"]["linewidth"],
            fliersize=cfg.plot_elements["marker_size"],
        )
        ax[i].get_legend().remove()
        ax[i].set(xlabel="", ylabel="", title=f"$c={coop}$")
        _add_grid(ax[i])
        _set_axis_style(ax[i])
        # sns.stripplot(
        #     data=pd.DataFrame(results[coop][m]),
        #     ax=ax[j, i],
        #     color=cfg.colors['bluegray_4'],
        #     size=cfg.plot_elements['markers']['size'],
        # )
        # if Y_LIM is not None:
        #     ax.set(ylim=yl)

    _set_heuristic_xtick_labels(ax, heuristics)
    ax[0].set(ylabel=cfg.labels[measure_success_label])
    ax[4].legend(
        loc="lower right",
        bbox_to_anchor=(1.75, 0.0),
        # bbox_to_anchor=(0.1, -0.05),
        fancybox=True,
        shadow=True,
        ncol=1,
    )

    # ylim = [np.inf, -np.inf]
    # for a in ax:
    #     ylim[0] = np.min([a.get_ylim()[0], ylim[0]])
    #     ylim[1] = np.max([a.get_ylim()[1], ylim[1]])
    # plt.setp(ax, ylim=ylim)
    savename = savepath.joinpath(
        f'asymmetric_task_success_dist_{str(d).replace(".","_")}.pdf'
    )
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_synergy_norm_by_c_lineplot(resultspaths, ax, d, c, target_variable):
    results = load_experiment_results(
        loadpath=Path(resultspaths[d])
        .resolve()
        .joinpath("results_single_trial_analysis"),
        target_variable=target_variable,
    )
    set_paper_layout()

    try:
        heuristics_used = list(results[0.0][SYN_MEASURE].keys())
    except KeyError as err:
        logging.info(
            "Unknown measure %s, available measures: %s",
            SYN_MEASURE,
            list(results[0.0].keys()),
        )
        print(err.args)
        raise

    heuristic_colors = [
        cfg.colors["bl"],
        cfg.colors["ego"],
        cfg.colors["social"],
        cfg.colors["coop"],
        cfg.colors["adapt"],
    ]
    for h, color in zip(heuristics_used, heuristic_colors):
        estimates_by_c = {}
        for i, coop in enumerate(c):
            estimates_by_c[coop] = results[coop][SYN_MEASURE][h]
        ax.errorbar(
            x=c,
            y=pd.DataFrame(estimates_by_c).mean(),
            yerr=pd.DataFrame(estimates_by_c).std(ddof=1),
            fmt="",  # plot error markers only, the line is plottled slightly thicker below
            capsize=2,
            elinewidth=0.5,
            color=color,
            label=h,
        )
        ax.plot(
            c,
            pd.DataFrame(estimates_by_c).mean(),
            color=color,
            linewidth=1,
            zorder=10,
        )

    heuristic_labels = [cfg.labels[c] for c in heuristics_used]
    ax.legend(labels=heuristic_labels)
    ax.set(ylabel=cfg.labels[SYN_MEASURE])

    sns.despine(offset=2, trim=True)


def plot_figure_syngergy_by_c_and_heuristic_lineplot(resultspaths, savepath):
    set_paper_layout()

    n_figure_cols = 6
    n_figure_rows = 3
    savename = savepath.joinpath("synergy_by_c_and_heuristic_lineplot.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            n_figure_cols * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )
    gs = ax[0, 0].get_gridspec()
    # remove the underlying axes
    for a in ax[0, :5]:
        a.remove()
    axbig = fig.add_subplot(gs[0, :5])
    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    d = 0.0
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = "any_food"
    print("Plot results for non-distractor condition")
    plot_synergy_norm_by_c_lineplot(
        resultspaths, axbig, d, c, target_variable=t
    )
    # h = ['MH_BASELINE', 'MH_EGOISTIC', 'MH_SOCIAL1', 'MH_COOPERATIVE', 'MH_ADAPTIVE']
    # d = 0.5
    # c = [0.1]
    h = ["MH_EGOISTIC", "MH_SOCIAL1", "MH_COOPERATIVE"]
    d = 0.2
    c = [0.8]
    t = "any_food"
    print(f"Plot results for distractor condition (d={d})")
    # plot_synergy_norm_by_c(resultspaths, ax[0, 5], d, c, heuristics=h, t=t)

    d = 0.0
    c = [0.25]
    h = ["MH_EGOISTIC", "MH_ADAPTIVE", "MH_COOPERATIVE"]
    t = "any_food"
    print("Plot results for synergy vs. task success")
    plot_synergy_vs_task_success(
        resultspaths, ax[1, 5], d=d, c=c, heuristics=h, t=t
    )

    # Plot correlation between synergy and joint actions.
    t = "any_food"
    h = ["MH_EGOISTIC", "MH_COOPERATIVE"]
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    plot_correlation_syn_coop_actions(
        resultspaths, ax=ax[1:, :5], d=d, c=c, heuristics=h, t=t
    )

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.2, right=0.9)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_pid_estimator_comparison_boxplot(resultspaths, savepath):
    """Plot lineplot comparison of synergy estimation using different estimators.

    Parameters
    ----------
    savepath : pathlib.Path
        Figure save path
    """
    d = 0.0
    target_variable = "any_food"
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    estimator_comparison = ["sx", "sx_cor", "iccs", "syndisc"]
    estimator_labels = {
        "syn_norm_sx": "$I^{Sx}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_sx_cor": "$I^{Sx^{c}}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_iccs": "$I^{CCS}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_syndisc": "$I^{SD}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
    }

    set_paper_layout()

    n_figure_cols = len(h)
    n_figure_rows = 4
    savename = savepath.joinpath("synergy_estimation_comparison_box.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            4 * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )

    for estimator, a in zip(estimator_comparison, ax):
        measure = "syn_norm_" + estimator
        plot_synergy_norm_by_c(
            resultspaths,
            a,
            d,
            c,
            heuristics=h,
            t=target_variable,
            measure=measure,
        )
        a[0].set(ylabel=estimator_labels[measure])

    _unify_ylim(ax.flatten())

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.2, right=0.9)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_pid_estimator_comparison_lineplot(resultspaths, savepath):
    """Plot lineplot comparison of synergy estimation using different estimators.

    Parameters
    ----------
    savepath : pathlib.Path
        Figure save path
    """
    d = 0.0
    target_variable = "any_food"
    c = [0.0, 0.25, 0.5, 0.75, 1.0]
    h = [
        "MH_BASELINE",
        "MH_EGOISTIC",
        "MH_SOCIAL1",
        "MH_COOPERATIVE",
        "MH_ADAPTIVE",
    ]
    estimator_comparison = ["sx", "sx_cor", "iccs", "syndisc"]
    estimator_labels = {
        "syn_norm_sx": "$I^{Sx}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_sx_cor": "$I^{Sx^{c}}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_iccs": "$I^{CCS}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
        "syn_norm_syndisc": "$I^{SD}_{syn}(F;A_0,A_1)/I(F; A_0,A_1)$",
    }

    results = load_experiment_results(
        loadpath=Path(resultspaths[d])
        .resolve()
        .joinpath("results_single_trial_analysis"),
        target_variable=target_variable,
    )
    set_paper_layout()

    n_figure_cols = 1
    n_figure_rows = 4
    savename = savepath.joinpath("synergy_estimation_comparison_line.pdf")

    fig, ax = plt.subplots(
        ncols=n_figure_cols,
        nrows=n_figure_rows,
        figsize=(
            len(h) * cfg.plot_elements["figsize"]["colwidth_in"] + 0.5,
            n_figure_rows * cfg.plot_elements["figsize"]["lineheight_in"],
        ),
    )

    try:
        heuristics_used = list(results[0.0][SYN_MEASURE].keys())
    except KeyError as err:
        logging.info(
            "Unknown measure %s, available measures: %s"
            % (SYN_MEASURE, list(results[0.0].keys()))
        )
        raise err

    heuristic_colors = [
        cfg.colors["bl"],
        cfg.colors["ego"],
        cfg.colors["social"],
        cfg.colors["coop"],
        cfg.colors["adapt"],
    ]

    for estimator, a in zip(estimator_comparison, ax):
        measure = "syn_norm_" + estimator

        for h, color in zip(heuristics_used, heuristic_colors):
            estimates_by_c = {}
            for coop in c:
                estimates_by_c[coop] = results[coop][measure][h]
            a.errorbar(
                x=c,
                y=pd.DataFrame(estimates_by_c).mean(),
                yerr=pd.DataFrame(estimates_by_c).std(ddof=1),
                fmt="",  # plot error markers only, the line is plottled slightly thicker below
                capsize=2,
                elinewidth=0.5,
                color=color,
                label=h,
            )
            a.plot(
                c,
                pd.DataFrame(estimates_by_c).mean(),
                color=color,
                linewidth=1,
                zorder=10,
            )
        a.set(ylabel=estimator_labels[measure])

    heuristic_labels = [cfg.labels[c] for c in heuristics_used]
    ax[0].legend(labels=heuristic_labels)
    ax[-1].set(xlabel="$c$")

    _unify_ylim(ax)

    sns.despine(offset=2, trim=True)

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.2, right=0.9)
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_local_sx_pid_examples(resultspaths, savepath):
    """Plot local SxPID for exemplary setups and trials"""
    set_paper_layout()
    plt.rc(
        "xtick", labelsize=cfg.plot_elements["textsize"]["large"]
    )  # fontsize of the tick labels
    d = 0.0
    c = 0.5
    h = "MH_COOPERATIVE"
    target_variable = "any_food"
    trial = 1
    df = pd.read_csv(
        Path(resultspaths[d]).joinpath(
            "results_single_trial_analysis",
            "local_sx_pid",
            f"local_sx_pid_{h}_c_{c:4.2f}_t_{target_variable}_trial_{trial}.csv",
        )
    )
    fig = plot_local_sx_pid(df)

    savename = savepath.joinpath(
        f"sx_pid_{h}_c_{c}_t_{target_variable}_trial_{trial}.pdf"
    )
    logging.info("Saving figure to %s", savename)
    plt.savefig(savename)
    plt.show()


def plot_local_sx_pid(df):
    """Plot local SxPID estimates for a single trial"""
    # fig_height = df.shape[0] * 0.8

    fig, ax = plt.subplots(
        ncols=7,
        # figsize=(7, fig_height),
        figsize=(
            cfg.plot_elements["figsize"]["maxwidth_in"],
            cfg.plot_elements["figsize"]["lineheight_in"] * df.shape[0] * 0.3,
        ),
        gridspec_kw={"width_ratios": [2, 1, 1, 1, 1, 1, 1]},
    )

    fmt_pid = ".3f"
    annot_font_size = 7
    # cmap_lpid = "bwr"  # "bwr", "coolwarm"
    # cmap_lpid = sns.diverging_palette(0, 255, center="light", as_cmap=True)
    # seaborn.diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center='light', as_cmap=False)
    cmap_lpid = sns.blend_palette(
        [cfg.colors["adapt"], ".99", cfg.colors["coop"]], 40
    )
    lpid_min = (
        df[
            [
                cfg.col_lmi_s1_s2_t,
                cfg.col_unq_s1,
                cfg.col_unq_s2,
                cfg.col_shd,
                cfg.col_syn,
            ]
        ]
        .min()
        .min()
    )
    lpid_max = (
        df[
            [
                cfg.col_lmi_s1_s2_t,
                cfg.col_unq_s1,
                cfg.col_unq_s2,
                cfg.col_shd,
                cfg.col_syn,
            ]
        ]
        .max()
        .max()
    )

    def _plot_heatmap(
        cols, x_labels, ax, cmap, vmin=lpid_min, vmax=lpid_max, annot_fmt=None
    ):
        if annot_fmt is None:
            annot_fmt = fmt_pid
        sns.heatmap(  # outcomes, s1, s2, t
            cols,
            ax=ax,
            cbar=False,
            cmap=cmap,
            annot=True,
            fmt=annot_fmt,
            annot_kws={"size": annot_font_size},
            vmin=vmin,
            vmax=vmax,
            center=0,
        )
        ax.set(xticklabels=x_labels)

    _plot_heatmap(  # outcomes, s1, s2, t
        df[[cfg.col_s1, cfg.col_s2, cfg.col_t]],
        x_labels=["$A_0$", "$A_1$", "$G$"],
        ax=ax[0],
        cmap="Pastel1_r",
        vmin=0,
        vmax=1,
        annot_fmt="",
    )
    _plot_heatmap(  # joint probability, p(s1,s2,t)
        df[[cfg.col_joint_prob]],
        x_labels=["$p(A_0, A_1, G)$"],
        ax=ax[1],
        cmap="Greys",
        vmin=0,
        vmax=df[cfg.col_joint_prob].max(),
    )
    _plot_heatmap(  # local joint MI, i(s1,s2,t)
        df[[cfg.col_lmi_s1_s2_t]],
        # df[[cfg.col_lmi_s1_s2_t, cfg.col_lmi_s1_t, cfg.col_lmi_s2_t]],
        x_labels=["$i(G; A_0, A_1)$"],
        ax=ax[2],
        vmin=df[[cfg.col_lmi_s1_s2_t]].min().min(),
        vmax=df[[cfg.col_lmi_s1_s2_t]].max().max(),
        cmap="Greys",
    )
    _plot_heatmap(  # Unq1
        df[[cfg.col_unq_s1]],
        x_labels=["$i_{unq}(G; A_0)$"],
        ax=ax[3],
        cmap=cmap_lpid,  # "Reds"
    )
    _plot_heatmap(  # Unq2
        df[[cfg.col_unq_s2]],
        x_labels=["$i_{unq}(G; A_1)$"],
        ax=ax[4],
        cmap=cmap_lpid,  # "Greens",
    )
    _plot_heatmap(  # Shd
        df[[cfg.col_shd]],
        x_labels=["$i_{shd}(G; A_0, A_1)$"],
        ax=ax[5],
        cmap=cmap_lpid,  # "Reds",
    )
    _plot_heatmap(  # Syn
        df[[cfg.col_syn]],
        x_labels=["$_{syn}(G; A_0, A_1)$"],
        ax=ax[6],
        cmap=cmap_lpid,  # "Blues",
    )
    for a in ax.flatten():
        a.xaxis.set_ticks_position("top")
        a.axes.get_yaxis().set_visible(False)
        a.set_xticklabels(a.get_xticklabels(), rotation=45)
        for _, spine in a.spines.items():
            spine.set_visible(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    return fig


def main(version):
    cfg.plot_elements["figsize"] = {  # overwrite figure size
        "maxwidth_cm": 12,  # \textwidth of standard LaTeX articles
        "maxwidth_in": 4.77,  # \textwidth of standard LaTeX articles
        "lineheight_in": 1.5,
        "colwidth_in": 0.88,  # max. width divided by 5
    }
    resultspaths = {
        0.0: f"../../lbf_experiments/shared_goal_dist_0_0_v{version}",
        0.2: f"../../lbf_experiments/shared_goal_dist_0_2_v{version}",
        0.5: f"../../lbf_experiments/shared_goal_dist_0_5_v{version}",
        "asym": f"../../lbf_experiments/asymmetric_d_0_0_v{version}",
    }
    initialize_logger(log_name="plot_paper_figures")
    savepath = Path(
        f"../../lbf_experiments/#journal_figures/v{version}_sign_{SIGNIFICANT_ONLY}"
    )
    savepath.mkdir(parents=True, exist_ok=True)

    if SIGNIFICANT_ONLY:
        print("Plotting only significant trials")

    # Plot main results on synergy as a measure of cooperation.
    plot_figure_syngergy_by_c_and_heuristic(resultspaths, savepath)
    # plot_figure_syngergy_by_c_and_heuristic_lineplot(resultspaths, savepath)

    # Plot asymmetric results.
    plot_figure_asymmetric_results(resultspaths, savepath)

    # Plot task performance/manipulation check.
    plot_figure_manipulation_check(resultspaths, savepath)

    # Plot local SxPID for selected trials
    plot_local_sx_pid_examples(resultspaths, savepath)

    # # Plot backup figures.
    plot_asymmetric_task_success(resultspaths, savepath)
    # plot_pid_estimator_comparison_lineplot(resultspaths, savepath)
    plot_pid_estimator_comparison_boxplot(resultspaths, savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Plot final results for paper")
    )
    parser.add_argument(
        "--version", "-v", type=int, default=9, help="Results version to use"
    )
    # version = 9  # same as 8 but using the fixed SxPID correction
    # version = 8  # patience=15, binary encoding of agent actions
    # version = 7  # patience=15, binary encoding of agent actions
    # version = 4  # patience=15
    # version = 5  # patience=50
    args = parser.parse_args()

    main(args.version)
