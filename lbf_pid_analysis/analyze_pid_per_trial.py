#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Estimate partial information decomposition for LBF game data.
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
import pickle
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

import idtxl.idtxl_utils as utils  # pylint: disable=E0401

from utils import initialize_logger, read_settings
from mi_estimation import (
    estimate_h_discrete,
    estimate_mi_discrete,
)
from pid_estimation import estimate_pid, estimate_joint_distribution
from plot_measure_by_c import set_layout
import config as cfg

N_AGENTS = 2
PID_MEASURES = ["syn", "shd", "unq1", "unq2", "coinfo", "coop_ratio"]
ESTIMATORS = ["sx", "sx_cor", "iccs", "broja", "syndisc"]


def summary_plot_pid(settings, results_by_heuristic, target_variable, filename, render):
    """Plot summary of estimated PID measures.

    Parameters
    ----------
    settings : dict
        Plotting settings
    results_by_heuristic : dict
        Results for various measures by heuristic and cooperation parameter
    target_variable : str
        Target variable used in estimation
    filename : patlib.Path
        Filename and -path to save output file to
    render : bool
        Whether to show generated plot after saving
    """
    measures = []
    for est in ESTIMATORS:
        measures.append(
            [f"syn_{est}", f"shd_{est}", f"unq1_{est}", f"unq2_{est}"],
        )

    colors = [
        [cfg.colors["blue"] for i in range(4)],
        [cfg.colors["yellow"] for i in range(4)],
        [cfg.colors["orange"] for i in range(4)],
        [cfg.colors["green"] for i in range(4)],
        [cfg.colors["bluegray_2"] for i in range(4)],
    ]

    labels = []
    for est in ["SX", "SX^c", "CCS", "BROJA", "SD"]:
        labels.append(
            [
                "$I_{syn}^{" + est + "}$",
                "$I_{shd}^{" + est + "}$",
                "$I_{unq}^{" + est + "}(F;A_0)$",
                "$I_{unq}^{" + est + "}(F;A_1)$",
            ]
        )

    fig = make_summary_plot(
        measures, colors, labels, results_by_heuristic, correlate_measure=None
    )
    plt.suptitle(
        "Info theory results for sight: {}, coop: {}, target: {}".format(  # pylint: disable=bad-option-value,C0209
            settings["environment"]["sight"],
            float(settings["environment"]["coop"]),
            target_variable,
        )
    )
    fig.tight_layout()
    logging.info("Saving figure to %s", filename)
    fig.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def summary_plot(settings, results_by_heuristic, target_variable, filename, render):
    """Plot summary of information-theoretic estimates.

    Parameters
    ----------
    settings : dict
        Plotting settings
    results_by_heuristic : dict
        Results for various measures by heuristic and cooperation parameter
    target_variable : str
        Target variable used in estimation
    filename : patlib.Path
        Filename and -path to save output file to
    render : bool
        Whether to show generated plot after saving
    """
    measures = [
        ["h", "jmi", "mi_iccs", "mi_sx"],
        ["syn_norm_iccs", "syn_norm_sx", "syn_norm_sx_cor", "syn_norm_syndisc"],
        [
            "total_food_value_collected",
            "any_food_collected",
            "n_cooperative_actions",
        ],
    ]
    colors = [
        [cfg.colors["blue"] for i in range(4)],
        [cfg.colors["yellow"] for i in range(4)],
        [cfg.colors["bluegray_2"] for i in range(3)],
    ]
    labels = [
        [
            "$H(F)$",
            "$I(A_0,A_1;F)$",
            "$I^{CCS}(A_0,A_1;F)$",
            "$I^{Sx}(A_0,A_1;F)$",
        ],
        [
            "$I_{syn}^{CCS}/I$",
            "$I_{syn}^{SX}/I$",
            "$I_{syn}^{SX^c}/I$",
            "$I_{syn}^{SD}/I$",
        ],
        [
            "total value collected",
            "N items collected",
            "# cooperative actions $M$",
        ],
    ]

    fig = make_summary_plot(
        measures,
        colors,
        labels,
        results_by_heuristic,
        correlate_measure="syn_sx",
    )
    plt.suptitle(
        "Info theory results for sight: {}, coop: {}, target: {}".format(  # pylint: disable=C0209,bad-option-value
            settings["environment"]["sight"],
            float(settings["environment"]["coop"]),
            target_variable,
        )
    )
    fig.tight_layout()
    logging.info("Saving figure to %s", filename)
    fig.savefig(filename)
    if render:
        plt.show()
    else:
        plt.close()


def make_summary_plot(  # pylint: disable=too-many-locals
    measures, colors, labels, results_by_heuristic, correlate_measure="syn_sx"
):
    """Plotting routine for generating box plots of estimates

    _extended_summary_

    Parameters
    ----------
    measures : iterable
        List of measures to plot
    colors : iterable
        List of colors used in each plot
    labels : iterable
        List of labels used in each plot
    results_by_heuristic : dict
        Estimates collected over experiments
    correlate_measure : str, optional
        Measure to use for correlation plot, by default "syn_sx"

    Returns
    -------
    matplotlib.figure
        Generated figure
    """
    fig, ax = plt.subplots(
        ncols=4, nrows=len(measures), figsize=(8, len(measures) * 1.5)
    )
    for measure, a, c, l in zip(measures, ax, colors, labels):
        for i, m in enumerate(measure):
            np.random.seed(0)  # to control placement of strip plot data points
            sns.boxplot(
                data=pd.DataFrame(results_by_heuristic[m]),
                ax=a[i],
                color=c[i],
                linewidth=1,
                fliersize=3,
            )
            sns.stripplot(
                data=pd.DataFrame(results_by_heuristic[m]),
                ax=a[i],
                color=cfg.colors["bluegray_4"],
                size=3,
            )
            a[i].set_ylabel(l[i], fontsize=8)
            # if Y_LIM is not None:
            #     ax.set(ylim=yl)

    heuristics_used = list(results_by_heuristic[measures[0][0]].keys())
    if correlate_measure is not None:
        set_layout()
        correlations = []
        for coop in heuristics_used:
            correlations.append(
                np.corrcoef(
                    results_by_heuristic[correlate_measure][coop],
                    results_by_heuristic["n_cooperative_actions"][coop],
                )[0, 1]
            )
        if np.isnan(correlations).any():  # replace NaNs by zeros for proper plotting
            correlations = np.array(correlations)
            correlations[np.isnan(correlations)] = 0
        ax[len(measures) - 1, 3].bar(
            np.arange(len(heuristics_used)),
            correlations,
            color=cfg.colors["bluegray_3"],
            edgecolor=cfg.colors["bluegray_4"],
        )
        ax[len(measures) - 1, 3].set(
            ylabel=f"$c(${cfg.labels[correlate_measure]}$, M)$"
        )
    elif len(c) == 3:
        ax[len(measures) - 1, 3].axis("off")

    heuristic_labels = [cfg.labels[c] for c in heuristics_used]
    sns.despine(offset=2)
    for a in ax.flatten():
        a.tick_params(axis="y", which="both", labelsize=8)
        a.xaxis.set_ticks(
            np.arange(len(heuristic_labels))
        )  # to avoid FixedFormatter warning
        a.set_xticklabels(
            heuristic_labels,
            rotation=35,
            ha="right",
            fontsize=7,
            rotation_mode="anchor",
        )
    return fig


def get_heuristic_used(settings):
    """Infer agent heuristic"""
    if settings["agents"]["heuristic"] == "MultiHeuristicAgent":
        return "MH_{}".format(settings["agents"]["abilities"])
    return "SH_{}".format(settings["agents"]["heuristic"])


def get_trial_file_names(foldername):
    """Return file names in folder"""
    trial_files = glob.glob(str(foldername.joinpath("*game_data*.csv")))
    if len(trial_files) == 0:
        raise FileNotFoundError(
            f"Did not find any results to analyze in folder {foldername}!"
        )
    logging.info(
        "Reading trial data from folder: %s,\nfound %d trials: %s",
        foldername,
        len(trial_files),
        trial_files[0],
    )
    return sorted(trial_files)


def load_experiment_data(filename, load_field=False):
    """Load behavioral data and environment field"""
    df = pd.read_csv(filename)
    logging.debug(df)
    assert len(df.agent_id.unique()) == N_AGENTS
    df = df.replace({"action": cfg.actions})  # pylint: disable=no-member
    if load_field:
        field = np.load(filename.replace("game_data", "field").replace("csv", "npy"))
        logging.debug(field)
        return df, field
    return df


def get_experiment_performance(  # pylint: disable=too-many-locals
    df, settings, trial, outpath, render
):
    """Return performance in individual trials"""
    food_value_0 = df[df.agent_id == 0].food.values
    food_type_0 = df[df.agent_id == 0].food_type.values
    food_collected_0 = (food_value_0 > 0).astype(int)
    food_value_1 = df[df.agent_id == 1].food.values
    food_type_1 = df[df.agent_id == 1].food_type.values
    food_collected_1 = (food_value_1 > 0).astype(int)
    logging.debug(
        "Total food value agent 0: %d - agent 1: %d",
        food_value_0.sum(),
        food_value_1.sum(),
    )

    assert (
        df.step.max() == settings["environment"]["max_episode_steps"] - 1
    ), "Unequal step sizes: {}, {}".format(
        df.step.max(), settings["environment"]["max_episode_steps"] - 1
    )

    assert np.all(
        df[df.agent_id == 0].cooperative_actions.values
        == df[df.agent_id == 1].cooperative_actions.values
    )
    cooperative_actions = df[df.agent_id == 0].cooperative_actions.values

    assert np.all(
        (df[df.agent_id == 0].cooperative_actions.values == 2)
        == (df[df.agent_id == 1].cooperative_actions.values == 2)
    )
    food_type = np.zeros(len(food_type_0), dtype=int)
    food_type[np.logical_or(food_type_0 == 1, food_type_1 == 1)] = 1
    food_type[np.logical_or(food_type_0 == 2, food_type_1 == 2)] = 2

    _, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 5))
    ax[0, 0].scatter(np.where(food_value_0)[0], food_value_0[food_value_0 > 0], c="b")
    ax[0, 0].set_title("Food Agent 0")
    ax[0, 0].set_ylim([0, ax[0, 0].get_ylim()[1] * 1.1])
    ax[0, 1].plot(np.cumsum(food_value_0), "b")
    ax[0, 1].set_title("Food Sum Agent 0")
    ax[0, 2].axis("off")
    ax[1, 0].scatter(np.where(food_value_1)[0], food_value_1[food_value_1 > 0], c="r")
    ax[1, 0].set_title("Food Agent 1")
    ax[1, 0].set_ylim([0, ax[1, 0].get_ylim()[1] * 1.1])
    ax[1, 1].plot(np.cumsum(food_value_1), "r")
    ax[1, 1].set_title("Food Sum Agent 1")
    ax[1, 2].axis("off")
    # ax[2, 0].plot(field.sum(axis=1).sum(axis=0), c=cfg.colors['gray'])
    # ax[2, 0].set_title("Sum field food level")
    # ax[2, 1].plot(np.count_nonzero(field, axis=0).sum(axis=0), c=cfg.colors['gray'])
    ax[2, 1].set_title("Field food count")
    ax[2, 2].plot(cooperative_actions, c=cfg.colors["gray"])
    ax[2, 2].set_title("Cooperative actions")
    plt.suptitle(
        "{} sight: {}, coop: {}".format(
            get_heuristic_used(settings),
            settings["environment"]["sight"],
            float(settings["environment"]["coop"]),
        )
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    if outpath is not None:
        plt.savefig(
            outpath.joinpath(
                "{}_c_{}_payoff_trial_{}.pdf".format(
                    get_heuristic_used(settings),
                    float(settings["environment"]["coop"]),
                    trial,
                )
            )
        )
    assert (
        np.sum(cooperative_actions)
        <= np.logical_or(food_collected_0, food_collected_1).sum()
    )

    # Avoid NaNs in output due to division by zero
    n_actions = np.logical_or(food_collected_0, food_collected_1).sum()
    if n_actions == 0:
        frac_cooperative_actions = 0.0
    else:
        frac_cooperative_actions = np.sum(cooperative_actions) / n_actions

    if render:
        plt.show()
    else:
        plt.close()

    return {
        "total_food_value_collected": np.sum(
            (food_value_0, food_value_1), axis=0
        ),  # total food value collected at any time step
        "mean_food_value_collected": np.mean(
            (food_value_0, food_value_1), axis=0
        ),  # mean food value collected at any time step
        "n_collections": food_collected_0
        + food_collected_1,  # no. food items collected by both agents jointly
        "food_value_collected_agent_0": food_value_0,  # food values collected by first agent
        "food_value_collected_agent_1": food_value_1,  # food values collected by second agent
        "n_collections_agent_0": food_collected_0,  # no. food items collected by first agent
        "n_collections_agent_1": food_collected_1,  # no. food items collected by second agent
        "any_food_collected": np.logical_or(food_collected_0, food_collected_1).astype(
            int
        ),  # if any agent collected an item
        "food_type_collected": food_type,  # type of food item collected, individual or joint goal
        "cooperative_actions": (cooperative_actions).astype(
            int
        ),  # if a cooperative action took place
        "frac_cooperative_actions": frac_cooperative_actions,  # return as fraction of all collections
    }


def encode_source_variables(experiment_data, source_type, source_encoding):
    """ "Discretize source variables that were requested in the settings.

    Parameters
    ----------
    experiment_data : pandas.DataFrame
        Experiment data
    source_type : str
        Which source to encode, can be
    source_encoding : str
        How to encode selected source, can be 'binary' (LOAD versus all other
        actions) or 'individual' (encodes every action individually)

    Returns
    -------
    numpy.ndarray
        Encoded source 1
    numpy.ndarray
        Encoded source 2
    """
    logging.debug(
        "Using agent %s as sources/inputs to PID analysis (encoding %s)",
        source_type,
        source_encoding,
    )

    if source_type == "actions":
        if source_encoding == "binary":
            actions_0 = np.zeros(
                len(experiment_data[experiment_data.agent_id == 0].action.values),
                dtype=int,
            )
            actions_0[
                experiment_data[experiment_data.agent_id == 0].action.values
                == cfg.actions["Action.LOAD"]
            ] = 1
            actions_1 = np.zeros(
                len(experiment_data[experiment_data.agent_id == 1].action.values),
                dtype=int,
            )
            actions_1[
                experiment_data[experiment_data.agent_id == 1].action.values
                == cfg.actions["Action.LOAD"]
            ] = 1
        elif source_encoding == "individual_actions":
            actions_0 = experiment_data[
                experiment_data.agent_id == 0
            ].action.values.astype(int)
            actions_1 = experiment_data[
                experiment_data.agent_id == 1
            ].action.values.astype(int)
        else:
            raise RuntimeError("Unknown source encoding %s" % source_encoding)
        actions_0_enc = np.squeeze(
            OrdinalEncoder().fit_transform(actions_0.reshape(-1, 1)).astype(int)
        )
        actions_1_enc = np.squeeze(
            OrdinalEncoder().fit_transform(actions_1.reshape(-1, 1)).astype(int)
        )
        assert len(np.unique(actions_0)) == len(np.unique(actions_0_enc))
        assert len(np.unique(actions_1)) == len(np.unique(actions_1_enc))
        source_0 = actions_0_enc
        source_1 = actions_1_enc

    elif source_type == "closest_distance":  # distance to closest target
        if source_encoding == "binary":
            raise RuntimeError(
                "Cannot use requested binary encoding to encode agent distances as sources"  # pylint: disable=C0301
            )
        dist_0 = experiment_data[
            experiment_data.agent_id == 0
        ].dist_closest_food.values.astype(int)
        dist_1 = experiment_data[
            experiment_data.agent_id == 1
        ].dist_closest_food.values.astype(int)
        dist_0_enc = np.squeeze(
            OrdinalEncoder().fit_transform(dist_0.reshape(-1, 1)).astype(int)
        )
        dist_1_enc = np.squeeze(
            OrdinalEncoder().fit_transform(dist_1.reshape(-1, 1)).astype(int)
        )
        source_0 = dist_0_enc
        source_1 = dist_1_enc

    else:
        raise RuntimeError(
            "Unknown source variable %s" % source_type  # pylint: disable=W1201,C0209
        )

    return source_0, source_1


def encode_target_variable(performance, target_variable):
    """ "Discretize target variable that was requested in the settings.

    Available target variables:
    - cooperative_actions
    - total_food_value_collected
    - mean_food_value_collected
    - any_food
    - food_type
    - n_collections
    - n_collections_agent_0
    - n_collections_agent_1
    """
    logging.debug("Using %s as target of PID analysis", target_variable)

    n_bins = 1
    if (
        target_variable == "cooperative_actions"
    ):  # whether a food item was collected collectively
        if len(np.unique(performance["cooperative_actions"])) == 1:
            target = np.zeros(len(performance["cooperative_actions"]), dtype=int)
        else:
            n_bins = 2
            target = utils.discretise(
                performance["cooperative_actions"], numBins=n_bins
            )
            assert len(np.unique(target)) <= 2
            assert np.max(target) <= 1
            assert np.min(target) >= 0

    elif target_variable == "mean_food_value_collected":  # mean collected food value
        if len(np.unique(performance["mean_food_value_collected"])) == 1:
            target = np.zeros(len(performance["mean_food_value_collected"]), dtype=int)
        else:
            n_bins = len(cfg.actions)
            target = utils.discretise(
                performance["mean_food_value_collected"], numBins=n_bins
            )

    elif (
        target_variable == "total_food_value_collected"
    ):  # total, summed collected food value
        if len(np.unique(performance["total_food_value_collected"])) == 1:
            target = np.zeros(len(performance["total_food_value_collected"]), dtype=int)
        else:
            n_bins = len(cfg.actions)
            target = utils.discretise(
                performance["total_food_value_collected"], numBins=n_bins
            )

    elif (
        target_variable == "n_collections"
    ):  # no. food items collected in each step, 0-2
        if len(np.unique(performance["n_collections"])) == 1:
            target = np.zeros(len(performance["n_collections"]), dtype=int)
        else:
            n_bins = 3
            target = utils.discretise(performance["n_collections"], numBins=n_bins)
            assert len(np.unique(target)) <= 3
            assert np.max(target) <= 2
            assert np.min(target) >= 0

    elif target_variable == "food_type":  # no. food items collected in each step, 0-2
        if len(np.unique(performance["food_type_collected"])) == 1:
            target = np.zeros(len(performance["food_type_collected"]), dtype=int)
        else:
            n_bins = 3
            target = utils.discretise(
                performance["food_type_collected"], numBins=n_bins
            )
            assert len(np.unique(target)) <= 3
            assert np.max(target) <= 2
            assert np.min(target) >= 0

    elif (
        target_variable == "n_collections_agent_0"
    ):  # no. food items collected in each step, 0-1
        if len(np.unique(performance["n_collections_agent_0"])) == 1:
            target = np.zeros(len(performance["n_collections_agent_0"]), dtype=int)
        else:
            n_bins = 2
            target = utils.discretise(
                performance["n_collections_agent_0"], numBins=n_bins
            )
            assert len(np.unique(target)) <= 2
            assert np.max(target) <= 1
            assert np.min(target) >= 0

    elif (
        target_variable == "n_collections_agent_1"
    ):  # no. food items collected in each step, 0-1
        if len(np.unique(performance["n_collections_agent_1"])) == 1:
            target = np.zeros(len(performance["n_collections_agent_1"]), dtype=int)
        else:
            n_bins = 2
            target = utils.discretise(
                performance["n_collections_agent_1"], numBins=n_bins
            )
            assert len(np.unique(target)) <= 2
            assert np.max(target) <= 1
            assert np.min(target) >= 0

    elif (
        target_variable == "any_food"
    ):  # whether any food item was collected in each step, 0-1
        if len(np.unique(performance["any_food_collected"])) == 1:
            target = np.zeros(len(performance["any_food_collected"]), dtype=int)
        else:
            n_bins = 2
            target = utils.discretise(performance["any_food_collected"], numBins=n_bins)
            assert len(np.unique(target)) <= 2
            assert np.max(target) <= 1
            assert np.min(target) >= 0

    else:
        raise RuntimeError("Unknown target variable {}".format(target_variable))

    # Additional ordinal encoding may be necessary if discretization led to empty bins
    # (i.e., number of unique values is smaller than the requested no. bins, in this
    # case, the method may put values in non-consecutive bins).
    target = np.squeeze(
        OrdinalEncoder().fit_transform(target.reshape(-1, 1)).astype(int)
    )

    return target, n_bins


def plot_variable_distributions(
    settings, source_0, source_1, target, outpath=None, figurename=None
):
    """Plot distribution of variables used in PID estimation"""
    fig, ax = plt.subplots(ncols=3, figsize=(6, 2))
    for a, var, color, label in zip(
        ax, [source_0, source_1, target], ["b", "b", "r"], ["s1", "s2", "t "]
    ):
        val, count = np.unique(var, return_counts=True)
        logging.info("\t%2s: %s - %s" % (label, val, count))  # pylint: disable=W1201
        bar = a.bar(val, count, color=color)
        a.bar_label(bar, label_type="edge")
        a.set(ylabel=f"{label} count")
    plt.suptitle(
        f'Variable distributions for {get_heuristic_used(settings)} sight: {settings["environment"]["sight"]}, '
        f'coop: {float(settings["environment"]["coop"])}'
    )
    fig.tight_layout()
    if outpath is not None:
        if figurename is None:
            raise RuntimeError("Provide a figurename to plot the variable distribution")
        fig.savefig(outpath.joinpath(figurename))
    plt.close("all")


def save_joint_distribution(source1, source2, target, outpath, savename):
    """Estimate joint distribution from time series and save to disk."""
    d = estimate_joint_distribution(source1, source2, target)
    pd.DataFrame(
        {
            "source1": [o[0] for o in d.outcomes],
            "source2": [o[1] for o in d.outcomes],
            "target": [o[2] for o in d.outcomes],
            "prob": [p for p in d.pmf],
        }
    ).to_csv(outpath.joinpath(savename))


def generate_output_path(path):
    """Generate output path for analysis results

    Create folder if it doesn't exist. Return full path.

    Parameters
    ----------
    path : str
        Parent folder containing experiment results

    Returns
    -------
    Path
        Path to results folder in parent directory
    """
    outpath = Path(path).joinpath("results_single_trial_analysis")
    Path(outpath).mkdir(parents=True, exist_ok=True)
    logging.info("Output path: %s", outpath)
    return outpath


def estimate_joint_mutual_information(source_0, source_1, target, n_perm, seed):
    """Estimate the joint mutual information between variables.

    Estimate the joint mutual information between the two sources and the
    target.

    Parameters
    ----------
    source_0 : numpy.ndarray
        Realizations of source 0
    source_1 : numpy.ndarray
        Realizations of source 1
    target : numpy.ndarray
        Realizations of the target
    n_perm : int, optional
        Number of permutations used in statistical test
    seed : int, optional
        Random seed for permutation testing, by default 1

    Returns target_entropy, mi, sign_mi, p
    -------
    float
        Target entropy
    float
        Joint mutual information
    bool
        If estimate is statistically significant (non-zero), given the specified
        alpha level
    float
        p-value of statistical test
    """
    target_entropy = estimate_h_discrete(target)
    if target_entropy == 0:
        logging.info("No entropy, skipping MI estimation and sign. test")
        mi = 0.0
        sign_mi = False
        p = 1.0
    else:
        mi, sign_mi, p = estimate_mi_discrete(
            np.vstack((source_0, source_1)).T,
            target,
            n_bins={"x1": len(cfg.actions), "x2": len(np.unique(target))},
            discretize_method="none",
            n_perm=n_perm,
            seed=seed,
        )
    logging.debug("Joint MI: %.4f (sign=%s, p=%.4f)", mi, sign_mi, p)
    return target_entropy, mi, sign_mi, p


def _append_pid_results(pid, est, results, heuristic):
    # Append PID estimates to global results structure
    results[f"syn_{est}"][heuristic].append(pid[cfg.col_syn])
    results[f"shd_{est}"][heuristic].append(pid[cfg.col_shd])
    results[f"unq1_{est}"][heuristic].append(pid[cfg.col_unq_s1])
    results[f"unq2_{est}"][heuristic].append(pid[cfg.col_unq_s2])
    results[f"coop_ratio_{est}"][heuristic].append(pid[cfg.col_coop_ratio])
    results[f"coinfo_{est}"][heuristic].append(pid[cfg.col_coinfo])
    results[f"mi_{est}"][heuristic].append(pid[cfg.col_mi])

    results[f"syn_norm_{est}"][heuristic].append(pid[cfg.col_syn_norm])
    results[f"shd_norm_{est}"][heuristic].append(pid[cfg.col_shd_norm])
    results[f"unq1_norm_{est}"][heuristic].append(pid[cfg.col_unq_s1_norm])
    results[f"unq2_norm_{est}"][heuristic].append(pid[cfg.col_unq_s2_norm])
    results[f"coop_ratio_norm_{est}"][heuristic].append(pid[cfg.col_coop_ratio_norm])
    results[f"coinfo_norm_{est}"][heuristic].append(pid[cfg.col_coinfo_norm])
    return results


def _append_trial_performance(results, heuristic, performance):
    # Collect aggregated behavioral results for plotting them later.
    results["total_food_value_collected"][heuristic].append(
        performance["total_food_value_collected"].sum()
    )
    results["any_food_collected"][heuristic].append(
        performance["any_food_collected"].sum()
    )
    results["n_collections"][heuristic].append(performance["n_collections"].sum())
    results["n_collections_agent_0"][heuristic].append(
        performance["n_collections_agent_0"].sum()
    )
    results["n_collections_agent_1"][heuristic].append(
        performance["n_collections_agent_1"].sum()
    )
    results["food_value_collected_agent_0"][heuristic].append(
        performance["food_value_collected_agent_0"].sum()
    )
    results["food_value_collected_agent_1"][heuristic].append(
        performance["food_value_collected_agent_1"].sum()
    )
    results["n_cooperative_actions"][heuristic].append(
        performance["cooperative_actions"].sum()
    )
    results["frac_cooperative_actions"][heuristic].append(
        performance["frac_cooperative_actions"]
    )
    return results


def main(path, settings_path, folders, target_variable, render=False):
    """Run PID estimation for trials in each experiment"""

    initialize_logger(
        log_name=f"analyze_experiment_results_{folders[0]}_to_{folders[-1]}_t_{target_variable}",
    )
    outpath = generate_output_path(path)

    results_all = {
        "total_food_value_collected": {},
        "any_food_collected": {},
        "n_collections": {},
        "n_collections_agent_0": {},
        "n_collections_agent_1": {},
        "food_value_collected_agent_0": {},
        "food_value_collected_agent_1": {},
        "n_cooperative_actions": {},
        "frac_cooperative_actions": {},
        "h": {},
        "jmi": {},
        "jmi_sign": {},
        "jmi_p": {},
        "mi_sources": {},
        "mi_sources_sign": {},
        "mi_sources_p": {},
        "source_corr": {},
    }
    for estimator in ESTIMATORS:
        for atom in PID_MEASURES:
            for norm in ["", "_norm"]:
                results_all[f"{atom}{norm}_{estimator}"] = {}
        results_all[f"mi_{estimator}"] = {}

    for folder_number in folders:  # each folder holds one experiment
        # Identify files collected for experiment. Load settings file.
        foldername = Path(path, f"{int(folder_number):02d}*")
        trial_files = get_trial_file_names(foldername)
        experiment_settings = read_settings(
            glob.glob(str(foldername.joinpath("experiment_settings.yml")))[0]
        )
        analysis_settings = read_settings(Path(settings_path))

        # Create data structures to collect data over trials.
        heuristic = get_heuristic_used(experiment_settings)
        for measure in results_all:
            results_all[measure][heuristic] = []
        logging.info(
            "Current experiment was run with heuristic %s sight: %d, coop: %.2f",
            heuristic,
            experiment_settings["environment"]["sight"],
            float(experiment_settings["environment"]["coop"]),
        )
        logging.info(
            "Target variable: %s, source variables: %s (encoding %s)",
            target_variable,
            analysis_settings["sources"],
            analysis_settings["source_encoding"],
        )

        Path(outpath).joinpath("trial_data").mkdir(parents=True, exist_ok=True)

        for trial in range(experiment_settings["experiment"]["ntrials"]):
            logging.info(
                "Analyzing folder %02d, heuristic %s, coop=%.2f - trial %02d ",
                int(folder_number),
                heuristic,
                float(experiment_settings["environment"]["coop"]),
                trial,
            )

            # Load experiment data, movements, field setup, performance.
            experiment_data = load_experiment_data(trial_files[trial])
            if trial < 5:
                op = outpath.joinpath("trial_data")
            else:
                op = None
            performance = get_experiment_performance(
                experiment_data,
                experiment_settings,
                trial,
                outpath=op,
                render=render,
            )
            results_all = _append_trial_performance(results_all, heuristic, performance)

            # Encode variables for information-theoretic analysis.
            source_0, source_1 = encode_source_variables(
                experiment_data,
                source_type=analysis_settings["sources"],
                source_encoding=analysis_settings["source_encoding"],
            )
            target, _ = encode_target_variable(performance, target_variable)
            if trial < 5:  # plot and save data for first n trials only
                plot_variable_distributions(
                    experiment_settings,
                    source_0,
                    source_1,
                    target,
                    outpath.joinpath("trial_data"),
                    figurename="{}_c_{}_{}_variable_dist_trial_{}.pdf".format(
                        heuristic,
                        float(experiment_settings["environment"]["coop"]),
                        target_variable,
                        trial,
                    ),
                )
                save_joint_distribution(
                    source_0,
                    source_1,
                    target,
                    outpath.joinpath("trial_data"),
                    savename="{}_c_{}_{}_variable_dist_trial_{}.csv".format(
                        heuristic,
                        float(experiment_settings["environment"]["coop"]),
                        target_variable,
                        trial,
                    ),
                )

            if np.all(source_0 == source_0[0]) or np.all(source_1 == source_1[0]):
                logging.info("Found constant value in one of the sources")

            for measure in ESTIMATORS:
                pid = estimate_pid(measure, source_0, source_1, target)
                results_all = _append_pid_results(pid, measure, results_all, heuristic)

            mi_sources, sign_mi_sources, p_mi_sources = estimate_mi_discrete(
                source_0,
                source_1,
                n_bins={"x1": len(cfg.actions), "x2": len(cfg.actions)},
                discretize_method="none",
                n_perm=analysis_settings["nperm"],
                seed=analysis_settings["seed"],
            )
            target_entropy, mi, sign_mi, p = estimate_joint_mutual_information(
                source_0,
                source_1,
                target,
                analysis_settings["nperm"],
                analysis_settings["seed"],
            )
            results_all["jmi"][heuristic].append(mi)
            results_all["jmi_sign"][heuristic].append(sign_mi)
            results_all["jmi_p"][heuristic].append(p)
            results_all["mi_sources"][heuristic].append(mi_sources)
            results_all["mi_sources_sign"][heuristic].append(sign_mi_sources)
            results_all["mi_sources_p"][heuristic].append(p_mi_sources)
            results_all["h"][heuristic].append(target_entropy)
            results_all["source_corr"][heuristic].append(
                np.corrcoef(source_0, source_1)[0, 1]
            )

    summary_plot_pid(
        experiment_settings,
        results_all,
        target_variable,
        filename=outpath.joinpath(
            f"pid_estimates_over_trials_{folders[0]}_to_{folders[-1]}_t_{target_variable}.pdf"
        ),
        render=render,
    )
    summary_plot(
        experiment_settings,
        results_all,
        target_variable,
        filename=outpath.joinpath(
            f"mi_estimates_over_trials_{folders[0]}_to_{folders[-1]}_t_{target_variable}.pdf"
        ),
        render=render,
    )
    _save_estimates(
        results_all,
        experiment_settings,
        filename=outpath.joinpath(
            f"mi_estimates_over_trials_{folders[0]}_to_{folders[-1]}_t_{target_variable}.p"
        ),
    )


def _save_estimates(results_all, experiment_settings, filename):
    results_all["experiment_settings"] = experiment_settings
    with open(filename, "wb") as f:
        pickle.dump(results_all, f, pickle.HIGHEST_PROTOCOL)
    logging.info("Results saved to %s\n\n", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PID estimation for LBF experiments.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "-p",
        "--path",
        default="../../lbf_experiments/shared_goal_dist_0_0_v10",
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
        help="List of folder numbers to analyze (e.g., all heuristics for a value of c)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="any_food",
        type=str,
        help=("Variable to use as target in PID estimation"),
    )
    args = parser.parse_args()

    main(args.path, args.settings, args.folders, args.target, args.render)
