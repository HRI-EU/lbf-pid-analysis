#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Estimate partial information decomposition for a series of logic gates and
# matrix/normal form games while varying input correlation.
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
from itertools import product
from pathlib import Path
from typing import Any

import config as cfg
import matplotlib
import numpy as np
import seaborn as sns
from dit import Distribution
from dit.shannon import mutual_information
from matplotlib import pyplot as plt
from pid_estimation import estimate_pid_from_dist
from plot_measure_by_c import unify_axis_ylim
from utils import initialize_logger

FIGTYPE = "svg"
R_PLOT = [-0.9, 0.0, 0.9]
# COLORS = ["#003653", "#af363c", "#246920", "#246920", "grey"]
# COLORS = ["#2678A5", "#C65359", "#3E9B50", "#3E9B50", "grey"]
COLORS = ["#0077BB", "#D94B55", "#52AA3E", "#52AA3E", "grey"]
# COLORS = [
#     cfg.colors["blue_muted"],
#     cfg.colors["red_muted"],
#     cfg.colors["green_muted"],
#     cfg.colors["green_muted"],
#     cfg.colors["grey_muted"],
# ]
PID_ATOMS = [cfg.col_syn, cfg.col_shd, cfg.col_unq_s1, cfg.col_unq_s2]


def estimate_source_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the plug-in mutual information I(X; Y) between sources in bits.

    Parameters
    ----------
    x : numpy.ndarray
        Realizations of first source
    y : numpy.ndarray
        Realizations of second source

    Returns
    -------
    float
        Mutual information between sources
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must contain the same number of observations.")

    if x.size == 0:
        raise ValueError("x and y must not be empty.")

    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("x and y must contain only 0 and 1.")

    observations = np.column_stack((x, y))
    outcomes, counts = np.unique(observations, axis=0, return_counts=True)
    probabilities = counts / counts.sum()

    # Outcomes can be tuples such as (0, 1).
    distribution = Distribution(
        [tuple(outcome) for outcome in outcomes],
        probabilities,
        rv_names=["X", "Y"],
    )

    return mutual_information(distribution, ["X"], ["Y"])


def correlated_gates(
    p: float, c: float, gate_name: str, N: int = 10000, eps: float = 0.1
) -> tuple[dict, float, int, float]:
    """Simulate gate distribution for correlated inputs


    Parameters
    ----------
    p : float
        Probability for taking action 1
    c : float
        Input correlation
    gate_name : str
        Name of gate to simulate
    N : int, optional
        Sample size, by default 10000
    eps : float, optional
        Noise parameter used by some gates, by default 0.1

    Returns
    -------
    dict
        Joint distribution
    float
        Actual correlation between inputs
    int
        Total reward (sum over target variable)
    float
        Mutual information between inputs
    """
    x, y = simulate_correlation(p, c, N)
    z = np.zeros(len(x))

    if gate_name == "and":
        z = np.logical_and(x, y).astype(int)
    elif gate_name == "or":
        z = np.logical_or(x, y).astype(int)
    elif gate_name == "xor":
        z = np.logical_xor(x, y).astype(int)
    elif gate_name == "unq1":
        z = x.copy()
    elif gate_name == "copy":
        z = np.zeros(len(x))
        z[np.logical_and(x == 0, y == 0)] = 0
        z[np.logical_and(x == 0, y == 1)] = 1
        z[np.logical_and(x == 1, y == 0)] = 2
        z[np.logical_and(x == 1, y == 1)] = 3
    elif gate_name == "andnot":
        z = np.zeros(len(x))
        z[np.logical_and(x == 0, y == 1)] = 1

    triplets, counts = np.unique(np.vstack((x, y, z)).T, axis=0, return_counts=True)
    outcomes = [(t[0], t[1], t[2]) for t in triplets]
    dist = generate_empty_dist()
    for o, p in zip(outcomes, counts / N):
        dist[o] = float(p)
    assert np.isclose(sum(dist.values()), 1.0), f"Total prob. != 1 in {dist}"
    assert (np.array(list(dist.values())) >= 0).all(), f"Prob. smaller zero in {dist}"

    mi_sources = estimate_source_mi(x, y)
    return dist, np.corrcoef(x, y)[0, 1], sum(z), mi_sources


def generate_empty_dist(
    n_s1_states: int = 2, n_s2_states: int = 2, n_target_states: int = 2
) -> dict:
    """Generate an empty dictionary with all possible joint realizations

    Parameters
    ----------
    n_s1_states : int, optional
        Number of states of source 1, by default 2
    n_s2_states : int, optional
        Number of states of source 2, by default 2
    n_target_states : int, optional
        Number of states of target variable, by default 2

    Returns
    -------
    dict
        Empty dictionary with all possible joint realizations as keys
    """
    a = [
        np.arange(n_s1_states),
        np.arange(n_s2_states),
        np.arange(n_target_states),
    ]
    realizations = list(product(*a))
    return {r: 0.0 for r in realizations}


def simulate_correlation(
    p: float, r: float, N: int = 1000
) -> tuple[
    np.ndarray[tuple[Any, ...], np.dtype[Any]],
    np.ndarray[tuple[Any, ...], np.dtype[Any]],
]:
    """Simulate two correlated binary variables.

    Following https://stackoverflow.com/a/65668646

    Parameters
    ----------
    p : float
        Probability of action 1
    r : float
        Correlation to be simulated
    N : int, optional
        Number of samples, by default 10000

    Returns
    -------
    numpy.ndarray
        Realizations of first variable
    numpy.ndarray
        Realizations of second variable
    """
    anticorrelation = False
    if r < 0.0:
        anticorrelation = True
    x = (np.random.rand(N) < p).astype(int)
    assert np.isclose(p, sum(x) / N, atol=0.05), (
        f"p_x - expected: {p}, actual: {sum(x) / N}"
    )
    if np.isclose(r, 0.0) or p == 0:
        y = (np.random.rand(N) < p).astype(int)
        assert np.isclose(p, sum(y) / N, atol=0.05), f"p={p}, p_sim={(sum(x) / N,)}"
    else:
        x_bar = np.mean(x)
        xy_bar = np.abs(r) * x_bar + (1 - np.abs(r)) * x_bar**2
        toflip = sum(x == 1) - round(len(x) * xy_bar)
        y = x.copy()
        y[np.random.choice(np.where(x == 0)[0], toflip, replace=False)] = 1
        y[np.random.choice(np.where(x == 1)[0], toflip, replace=False)] = 0
        if anticorrelation:
            # Invert y to generate anti-correlated variables.
            y = 1 - y
            assert np.isclose(1 - p, sum(y) / N, atol=0.05), (
                f"p_y - expected: {1 - p}, actual: {sum(y) / N}"
            )

        else:
            assert np.isclose(p, sum(y) / N, atol=0.05), (
                f"p_y - expected: {p}, actual: {sum(y) / N}"
            )

        # c = np.corrcoef(x, y)[0, 1]
        # assert np.isclose(c, r, atol=0.05), f"r={r} != c={c} (for p={p})"

    return x, y


def plot_input_correlation(p_0: float, savepath: Path) -> None:
    """Plot joint input distribution for some correlation coefficients"""
    fig, ax = plt.subplots(ncols=5, tight_layout=True, figsize=(8, 1.5))
    for r, a in zip([-1.0, -0.5, 0.0, 0.5, 1.0], ax):
        x, y = simulate_correlation(p_0, r)
        # h = a.hist2d(x, y, bins=2, cmap="Blues", density=True)

        hist, xedges, yedges = np.histogram2d(x, y, bins=2, density=False)
        im = a.imshow(
            hist / len(x),
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="Blues",
            interpolation="none",
            vmin=0,
            vmax=p_0,
        )
        fig.colorbar(im, ax=a, shrink=0.75)

        a.set(
            title=rf"r={r}",
            xticks=[0.25, 0.75],
            yticks=[0.25, 0.75],
            xticklabels=[0, 1],
            yticklabels=[0, 1],
        )
        # Minor ticks
        a.set_xticks([0.5], minor=True)
        a.set_yticks([0.5], minor=True)
        a.grid(which="minor", color="k", linestyle="-", linewidth=1)
        # fig.colorbar(h[3], ax=a)
    fig.savefig(
        savepath.joinpath(f"corr_gate_comparison_p_{p_0}_joint_input_dist.{FIGTYPE}")
    )


def plot_sanity_checks(
    simulated_c: list[float], r: list[float], savepath: Path
) -> None:
    """Plot simulated versus actual correlation as a sanity check"""
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    sns.lineplot(x=r, y=simulated_c, color="k", label="actual", ax=ax)
    sns.lineplot(x=r, y=r, color="grey", linestyle=":", label="expected", ax=ax)
    ax = plt.gca()
    ax.set(
        xticks=np.arange(r[0], r[-1] + 0.2, 0.2),
        yticks=np.arange(r[0], r[-1] + 0.2, 0.2),
        xlabel="$\\rho$",
        ylabel="$c$",
    )
    plt.legend()
    plt.tight_layout()
    fig.savefig(savepath.joinpath(f"sanity_check_sim_correlation.{FIGTYPE}"))


def _pid_line_plots(
    pid_full: dict,
    r: list[float],
    ax: matplotlib.axes.Axes,
    title: str | None = None,
    normalize: bool = False,
) -> None:
    """Plot PID over full range of input correlations"""
    measures = PID_ATOMS + ["mi"]
    styles = ["-", "-", "--", ":", "-"]

    # Plot estimates.
    if normalize:
        m_suffix = "_norm"
    else:
        ylabel = "PID"
        m_suffix = ""

    for c, s, m in zip(COLORS, styles, measures):
        sns.lineplot(
            x=r,
            y=np.array(pid_full[f"{m}{m_suffix}"]),
            color=c,
            label=m,
            linestyle=s,
            linewidth=1.7,
            legend=False,
            ax=ax,
        )
    ax.set(
        xlabel="$\\rho$",
        # xticks=[r[0], r[(len(r) - 1) // 2], r[-1]],
        xticks=[r[0], r[5], r[10], r[15], r[-1]],
        title=title,
    )
    ax.axhline(0, zorder=0, color="lightgray", linestyle=":", linewidth=1)


def _append_results(
    pid_full: dict, pid: dict, c: float, reward: int, mi_sources: float
) -> dict:
    for atom in PID_ATOMS:
        pid_full[atom].append(pid[atom])
        pid_full[f"{atom}_norm"].append(pid[f"{atom}_norm"])
    pid_full["mi"].append(pid[cfg.col_mi])
    pid_full["mi_norm"].append(pid[cfg.col_mi])  # add for more convenient plotting
    pid_full["mi_sources"].append(mi_sources)
    pid_full["corr"].append(c)
    pid_full["reward"].append(reward)
    return pid_full


def main(path: str | Path, p_0: float, estimators: list[str], seed: int) -> None:
    """Run and plot simulation of logic gates for simulated inputs

    Parameters
    ----------
    path : str
        Output path
    p_0 : float
        Probability of s_0=s_1=1
    estimators : list[str]
        List of PID estimators to use
    seed : int
        Random seed
    """
    initialize_logger(log_name="logic_gate_pid_comparison")
    savepath = Path(path).joinpath(
        "results_single_trial_analysis", "lbf_logic_gate_pid_comparison"
    )
    savepath.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # Rdn and Unq1 as used in dit:
    gates = ["and", "xor", "unq1", "copy"]
    corr_coefficients = [float(i) for i in np.arange(-1.0, 1.1, 0.1)]
    dist_all = []

    n_rows = 2 * len(estimators) + 2
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=len(gates),
        figsize=(
            cfg.plot_elements["max_width_in"] * 0.9,
            n_rows * cfg.plot_elements["row_height"],
        ),
    )

    for g, gate_name in enumerate(gates):
        e = 0
        for estimator in estimators:
            pid_full: dict[str, list] = {
                k: []
                for k in [  # add for more convenient plotting
                    "mi",
                    "mi_norm",
                    "mi_sources",
                    "corr",
                    "reward",
                ]
            }
            for atom in PID_ATOMS:
                pid_full[atom] = []
                pid_full[f"{atom}_norm"] = []

            for corr in corr_coefficients:
                dist, c, reward, mi_sources = correlated_gates(p_0, corr, gate_name)
                dist_all.append(dist)
                pid = estimate_pid_from_dist(estimator, dist)
                pid_full = _append_results(pid_full, pid, c, reward, mi_sources)

            # Add reward for OR gate
            if "and" in gates:
                pid_full["reward_or"] = []
                for corr in corr_coefficients:
                    _, _, reward, _ = correlated_gates(p_0, corr, gate_name="or")
                    pid_full["reward_or"].append(reward)

            if e == 0:
                title = gate_name.upper()
            else:
                None
            _pid_line_plots(pid_full, r=corr_coefficients, ax=ax[e, g], title=title)
            _pid_line_plots(
                pid_full, r=corr_coefficients, ax=ax[e + 1, g], normalize=True
            )
            ax[e, g].set(ylabel=cfg.labels[estimator].replace("G", "T"))
            ax[e + 1, g].set(ylabel=cfg.labels[f"norm_{estimator}"].replace("G", "T"))
            e += 2

        # Plot source correlation
        ax[e, g].plot(corr_coefficients, pid_full["mi_sources"], color="k")
        ax[e, g].set(xlabel=r"$\rho$")

        # Plot 'reward', i.e., no. ones in output.
        ax[e + 1, g].plot(corr_coefficients, pid_full["reward"], color="k")
        if gate_name == "and":
            ax[e + 1, g].plot(corr_coefficients, pid_full["reward_or"], "k--")
            ax[e + 1, g].legend(["AND", "OR"])
        ax[e + 1, g].set(xlabel=r"$\rho$")
        if g == 0:
            ax[e, g].set(ylabel=r"$I(A_1;A_2)$")
            ax[e + 1, g].set(ylabel=r"$\sum T$")

    unify_axis_ylim(ax[:-2, :-1])  # unify PID plots, except for copy gate
    unify_axis_ylim(ax[-1, :-1])  # unify reward plots, except for copy gate
    # ax[0, -1].set(ylim=[ax[0, 0].get_ylim()[0], ax[0, -1].get_ylim()[1] * 1.1])
    ax[-1, -1].set(ylim=[0, ax[-1, -1].get_ylim()[1] * 1.1])

    # leg = ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4)
    # leg.set_in_layout(False)
    handles, labels = ax[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        [cfg.labels[l].replace("G", "T") for l in labels],
        loc="upper left",
        bbox_to_anchor=(ax[0, 0].get_position().x0, 0.99),
        bbox_transform=fig.transFigure,
        ncol=5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))  # keep top 10 % for the legend

    logging.info("Saving figures to %s", savepath)
    fig.savefig(
        savepath.joinpath(
            f"corr_gate_comparison_p_{p_0}_{'_'.join(estimators)}.{FIGTYPE}"
        )
    )
    plot_input_correlation(p_0, savepath)
    plot_sanity_checks(pid_full["corr"], corr_coefficients, savepath)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Estimate and plot PID for logic gates/normal form games and correlated inputs"
        )
    )
    parser.add_argument("--path", type=str, default="../", help="Output path")
    parser.add_argument(
        "--estimators",
        "-e",
        nargs="+",
        type=str,
        default=["sx_cor"],
        help="PID estimator(s) to use",
    )
    parser.add_argument("--seed", "-s", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--prob", "-p", type=float, default=0.5, help="Probability of taking action 1"
    )
    args = parser.parse_args()
    main(args.path, args.prob, args.estimators, args.seed)
