#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Perform Bayesian group comparisons on PID estimates for different heuristics.
# Follows this tutorial:
# https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html
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

import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import arviz as az
import pymc3 as pm


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
        log_name=f"compare_heuristics_bayes_{folders[0]}_to_{folders[-1]}"
    )
    loadpath = Path(path).joinpath("results_single_trial_analysis")
    settings = read_settings(Path(settings_path))
    figurepath = loadpath.joinpath(
        f"bayes_results_{folders[0]}_to_{folders[-1]}"
    )
    figurepath.mkdir(parents=True, exist_ok=True)

    compare_pid_between_heuristics_bayes(
        folders,
        loadpath,
        savename=loadpath.joinpath(
            f"compare_heuristics_bayes_{folders[0]}_to_{folders[-1]}.csv"
        ),
        figurepath=figurepath,
        settings=settings,
        render=render,
    )


def compare_pid_between_heuristics_bayes(
    folders, loadpath, savename, figurepath, settings, render=False
):
    """Perform Bayesian statistical test to compare estimates for one heuristic against the other.

    Test follows the Python implementation of the example in Kruschke (2012):
    https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/BEST.html
    With some inspiration from https://github.com/treszkai/best/blob/master/best/plot.py
    on styling the figures.

    Available heuristics:
        "MH_BASELINE": "BL",
        "MH_EGOISTIC": "EGO",  # former H1
        "MH_SOCIAL1": "SOC1",  # former H2
        "MH_SOCIAL2": "SOC2",  # former H5
        "MH_COOPERATIVE": "COOP",  # value function, coop
        "MH_ADAPTIVE": "ADAPT",  # value function, coop or egoistic

    Parameters
    ----------
    folders : iterable
        Folders over which to run comparison
    loadpath : pathlib.Path
        Path to load estimates from
    savename : pathlib.Path
        Save path for results overview
    figurepath : pathlib.Path
        Save path for visual representation Bayesian analysis results
    settings : dict
        Analysis settings containing the random seed ('seed')
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
            if (x1 == 0).all() and (x2 == 0).all():
                logging.info("All data are zero, not running Bayesian test")
                res_bayes = pd.DataFrame(
                    {
                        "mean": np.zeros(2),
                        "sd": np.zeros(2),
                        "hdi_2.5%": np.zeros(2),
                        "hdi_97.5%": np.zeros(2),
                    },
                    index=["difference of means", "effect size"],
                )
            else:
                res_bayes = bayes_test(
                    x1,
                    x2,
                    comparison[0],
                    comparison[1],
                    measure,
                    settings["seed"],
                    figurepath,
                    render,
                )
            stats.append(
                {
                    "measure": f"{measure}_{target_variable}",
                    "group1": comparison[0],
                    "group2": comparison[1],
                    "diff_means_mean": res_bayes.loc["difference of means"][
                        "mean"
                    ],
                    "diff_means_sd": res_bayes.loc["difference of means"]["sd"],
                    "diff_means_hdi3%": res_bayes.loc["difference of means"][
                        "hdi_2.5%"
                    ],
                    "diff_means_hdi97%": res_bayes.loc["difference of means"][
                        "hdi_97.5%"
                    ],
                    "mean_nonzero": np.logical_or(
                        res_bayes.loc["difference of means"]["hdi_2.5%"] > 0,
                        res_bayes.loc["difference of means"]["hdi_97.5%"] < 0,
                    ),
                    "effect_size_mean": res_bayes.loc["effect size"]["mean"],
                    "effect_size_sd": res_bayes.loc["effect size"]["sd"],
                    "effect_size_hdi3%": res_bayes.loc["effect size"][
                        "hdi_2.5%"
                    ],
                    "effect_size_hdi97%": res_bayes.loc["effect size"][
                        "hdi_97.5%"
                    ],
                    "es_outside_rope": np.logical_or(
                        res_bayes.loc["effect size"]["hdi_2.5%"] > 0.1,
                        res_bayes.loc["effect size"]["hdi_97.5%"] < -0.1,
                    ),
                }
            )
    stats = pd.DataFrame(stats)
    stats.to_csv(savename, index=False)
    print(stats)


def bayes_test(y1, y2, label1, label2, metric, seed, figurepath, render=False):
    """Perform Bayesian group comparison for two groups.

    This analysis follows an example in the PyMC3 tutorials:
    https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html

    References
    ----------
    - J.K. Kruschke. Bayesian estimation supersedes the t-test. Journal of
      Experimental Psychology: General, 142(2):573–603, 2013.
      URL: https://doi.org/10.1037/a0029146

    Parameters
    ----------
    y1 : numpy.ndarray
        Realizations of group 1
    y2 : numpy.ndarray
        Realizations of group 2
    label1 : str
        Description of group 1
    label2 : str
        Description of group 2
    metric : str
        Metric that is compared as dependent variable
    seed : int
        Random seed
    figurepath : pathlib.Path
        Path to save results figures to
    render : bool, optional
        Whether to display generated figures, by default False

    Returns
    -------
    pandas.DataFrame
        Results of Bayesian comparison
    """
    file_prefix = (
        f"{metric}_{cfg.labels[label1].lower()}_vs_{cfg.labels[label2].lower()}"
    )
    np.random.seed(seed)

    df1 = pd.DataFrame({metric: y1, "heuristic": cfg.labels[label1]})
    df2 = pd.DataFrame({metric: y2, "heuristic": cfg.labels[label2]})
    y = pd.concat([df1, df2]).reset_index()
    # y[metric] = (y[metric] - y[metric].mean()) / y[metric].std()  # normalize data
    make_distribution_plot(
        y, metric, savename=figurepath.joinpath(f"{file_prefix}_data_dist.pdf")
    )

    mu_m = y[metric].mean()
    mu_s = y[metric].std() * 2
    # Define prior of the standard deviation according to Kruschke's paper.
    # This deviates from the pyMC3 BERT examples as this is more specific to
    # IQ values and their known distribution in the general population.
    sigma_low = y[metric].std() / 1000
    sigma_high = y[metric].std() * 1000
    # Don't use uniform priors.
    # sd_m = y[metric].std()
    # sd_s = y[metric].std() * 10

    with pm.Model() as model:
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)
        group1_std = pm.Uniform("group1_std", lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform("group2_std", lower=sigma_low, upper=sigma_high)
        # group1_std = pm.Normal("group1_std", mu=sd_m, sigma=sd_s)
        # group2_std = pm.Normal("group2_std", mu=sd_m, sigma=sd_s)
        v = pm.Exponential("v_minus_one", 1 / 29.0) + 1

    with model:  # pylint: disable=E1129
        lamb_1 = group1_std**-2
        lamb_2 = group2_std**-2
        group1 = pm.StudentT(
            "drug", nu=v, mu=group1_mean, lam=lamb_1, observed=y1
        )
        group2 = pm.StudentT(
            "placebo", nu=v, mu=group2_mean, lam=lamb_2, observed=y2
        )

        diff_of_means = pm.Deterministic(
            "difference of means", group2_mean - group1_mean
        )
        diff_of_stds = pm.Deterministic(
            "difference of stds", group2_std - group1_std
        )
        effect_size = pm.Deterministic(
            "effect size",
            diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2),
        )

    with model:  # pylint: disable=E1129
        trace = pm.sample(
            10000, return_inferencedata=True  # , cores=1
        )  # not working in Windows https://discourse.pymc.io/t/error-during-run-sampling-method/2522

    make_trace_plot(
        trace, savename=figurepath.joinpath(f"{file_prefix}_traceplot.pdf")
    )
    make_posterior_distribution_plot(
        trace,
        df1[metric],
        df2[metric],
        cfg.labels[label1],
        cfg.labels[label2],
        savename=figurepath.joinpath(f"{file_prefix}_posterior_dist.pdf"),
    )
    make_forest_plot(
        trace,
        savename=figurepath.joinpath(f"{file_prefix}_forest_plot.pdf"),
    )

    if render:
        plt.show()
    else:
        plt.close("all")

    res = az.summary(
        trace,
        hdi_prob=0.95,
        var_names=["difference of means", "difference of stds", "effect size"],
    )
    logging.info(res)
    return res


def make_distribution_plot(y, metric, savename):
    """Plot distribution of raw data, colored by group.

    Parameters
    ----------
    y : pandas.DataFrame
        Joint data
    metric : str
        Metric to use from data frame
    savename : pathlib.Path
        File name to save figure to
    """
    sns.histplot(data=y, x=metric, hue="heuristic", bins=20)
    plt.savefig(savename)


def make_trace_plot(trace, savename):
    """Plot trace.

    Parameters
    ----------
    trace : pymc3.MultiTrace
        Trace used in parameter estimation
    savename : pathlib.Path
        File name to save figure to
    """
    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig(savename)


def make_posterior_distribution_plot(trace, x1, x2, group1, group2, savename):
    """Plot posterior distributions.

    Parameters
    ----------
    trace : pymc3.MultiTrace
        Trace used in parameter estimation
    x1 : numpy.ndarray
        Data from group 1
    x2 : numpy.ndarray
        Data from group 2
    group1 : str
        Name of group 1
    group2 : str
        Name of group 2
    savename : pathlib.Path
        File name to save figure to
    """
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))
    hist_kwargs = {"linewidth": 2, "edgecolor": "w"}

    # Plot estimated parameter values for input data.
    az.plot_posterior(
        trace,
        ax=ax[1:],
        color=cfg.colors["histogram"],
        hdi_prob=0.95,
        kind="hist",
        bins=20,
        var_names=[
            "group1_mean",
            "group2_mean",
            "group1_std",
            "group2_std",
            "v_minus_one",
        ],
        figsize=(10, 8),
        textsize=10,
        point_estimate="mean",  # mean/median/mode
        **hist_kwargs,
    )
    # Plot estimated differences in parameter values.
    az.plot_posterior(
        trace,
        ax=ax[0, 2:],
        color=cfg.colors["histogram_dark"],
        hdi_prob=0.95,
        kind="hist",
        bins=20,
        var_names=["difference of means", "difference of stds", "effect size"],
        ref_val=0,  # display the percentage below and above the values in ref_val
        figsize=(10, 4),
        textsize=10,
        point_estimate="mean",  # mean/median/mode
        **hist_kwargs,
    )
    posterior_predictive_check(trace, x1, x2, ax=ax[0, :2])
    ax[0, 0].set_title(group1)
    ax[0, 1].set_title(group2)

    plt.tight_layout()
    fig.savefig(savename)


def make_forest_plot(trace, savename, hdi=0.95):
    """Plot forest plot of estimates using the 95 % HDI.

    Parameters
    ----------
    trace : pymc3.MultiTrace
        Trace used in parameter estimation
    savename : pathlib.Path
        File name to save figure to
    """
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    az.plot_forest(
        trace, var_names=["group1_mean", "group2_mean"], hdi_prob=hdi, ax=ax[0]
    )
    az.plot_forest(
        trace, var_names=["group1_std", "group2_std"], hdi_prob=hdi, ax=ax[1]
    )
    az.plot_forest(trace, var_names=["v_minus_one"], hdi_prob=hdi, ax=ax[2])
    plt.tight_layout()
    fig.savefig(savename)


def posterior_predictive_check(trace, x1, x2, ax):
    """Perform posterior predictive check

    See
    https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/diagnostics_and_criticism/posterior_predictive.html

    Parameters
    ----------
    trace : pymc3.MultiTrace
        Trace used in parameter estimation
    x1 : numpy.ndarray
        Data from group 1
    x2 : numpy.ndarray
        Data from group 2
    ax : iterable of matplotlib axes
        Axes to plot into, expected to be of length 2
    """
    hist_kwargs = {
        "edgecolor": "w",
        "linewidth": 1.2,
        "facecolor": cfg.colors["histogram_data"],
        "density": True,
        "bins": 20,
        "label": "Observation",
    }
    ax[0].hist(x1, **hist_kwargs)
    ax[1].hist(x2, **hist_kwargs)
    # Plot credible t-distributions.
    chain = 1
    n_curves = 50
    n_samples = len(trace["posterior"]["group1_mean"][chain])
    idxs = np.random.choice(np.arange(n_samples), n_curves, replace=False)
    for a, group, group_data in zip(ax, ["group1", "group2"], [x1, x2]):
        means = trace["posterior"][f"{group}_mean"][chain]
        sigmas = trace["posterior"][f"{group}_std"][chain]
        nus = trace["posterior"]["v_minus_one"][chain]
        xmin = np.min(group_data)
        xmax = np.max(group_data)
        xmin -= (xmax - xmin) * 0.05
        xmax += (xmax - xmin) * 0.05
        x = np.linspace(xmin, xmax, 1000)
        kwargs = {
            "color": cfg.colors["histogram_dark"],
            "zorder": 1,
            "alpha": 0.3,
        }
        for i in idxs:
            a.plot(x, st.t.pdf(x, nus[i] ** 2, means[i], sigmas[i]), **kwargs)

        a.text(
            0.95,
            0.95,
            r"$\mathrm{N}=%d$" % len(group_data),  # pylint: disable=C0209
            transform=a.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
        for loc in ["top", "right"]:
            a.spines[loc].set_color("none")  # don't draw
        a.spines["left"].set_color("gray")
        a.set(
            xlabel="Observation",
            ylabel="Probability",
            xlim=(xmin, xmax),
            yticks=[],
            ylim=0,
        )

    hist_kwargs.update({"label": None, "zorder": 3, "alpha": 0.3})
    ax[0].hist(x1, **hist_kwargs)
    ax[1].hist(x2, **hist_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare information-theoretic estimates of LBF experiments "
            "between heuristics."
        )
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
        help=(
            "List of folder numbers to analyze (e.g., all heuristics for one "
            "value of c)"
        ),
    )
    args = parser.parse_args()

    main(args.path, args.settings, args.folders, args.render)
