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
from pathlib import Path
from itertools import product

from matplotlib import pyplot as plt
import numpy as np
from prettytable import PrettyTable

from pid_estimation import estimate_pid_from_dist
from estimate_local_pid import calculate_weighted_local_pid
from plot_local_sx_pid_per_trial import plot_local_pid_stats
from plot_measure_by_c import unify_axis_ylim
from utils import initialize_logger
import config as cfg

FIGTYPE = "pdf"
R_PLOT = [-0.9, 0.0, 0.9]
COLORS = ["#003653", "#af363c", "#246920", "#246920", "grey"]

small = 8
medium = 10
large = 12
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
plt.rc("legend", fontsize=cfg.plot_elements["textsize"]["medium"])  # legend fontsize
plt.rc(
    "figure", titlesize=cfg.plot_elements["textsize"]["large"]
)  # fontsize of the figure title


def correlated_gates(p, c, gate_name, N=10000, eps=0.1):
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
    """
    x, y = simulate_correlation(p, c, N)

    if gate_name == "and":
        z = np.logical_and(x, y).astype(int)
    elif gate_name == "or":
        z = np.logical_or(x, y).astype(int)
    elif gate_name == "xor":
        z = np.logical_xor(x, y).astype(int)
    elif gate_name == "noisy_xor":
        z = np.logical_xor(x, y).astype(int)
        mask_0 = np.where(np.logical_and(x == 0, y == 0))[0]
        mask_1 = np.where(np.logical_and(x == 1, y == 1))[0]
        n_to_flip = int((len(mask_0) + len(mask_1)) * eps / 2)
        z[np.random.choice(mask_0, n_to_flip)] = 1
        z[np.random.choice(mask_1, n_to_flip)] = 1
    elif gate_name == "unq1":
        z = x.copy()
    elif gate_name == "copy":
        z = np.zeros(len(x))
        z[np.logical_and(x == 0, y == 0)] = 0
        z[np.logical_and(x == 0, y == 1)] = 1
        z[np.logical_and(x == 1, y == 0)] = 2
        z[np.logical_and(x == 1, y == 1)] = 3
    elif gate_name == "prisoner":
        z = np.logical_xor(x, y).astype(int)
        z[np.logical_and(x == 0, y == 0)] = 2
    elif gate_name == "stag":
        z = np.logical_and(x, y).astype(int)
        z[np.logical_and(x == 0, y == 0)] = 1

    triplets, counts = np.unique(np.vstack((x, y, z)).T, axis=0, return_counts=True)
    outcomes = [(t[0], t[1], t[2]) for t in triplets]
    dist = generate_empty_dist()
    for o, p in zip(outcomes, counts / N):
        dist[o] = float(p)
    assert np.isclose(sum(dist.values()), 1.0), f"Total prob. != 1 in {dist}"
    assert (np.array(list(dist.values())) >= 0).all(), f"Prob. smaller zero in {dist}"
    return dist, np.corrcoef(x, y)[0, 1], sum(z)


def generate_empty_dist(n_s1_states=2, n_s2_states=2, n_target_states=2):
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


def simulate_correlation(p, r, N=1000):
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
    assert np.isclose(
        p, sum(x) / N, atol=0.05
    ), f"p_x - expected: {p}, actual: {sum(x) / N}"
    if r == 0 or p == 0:
        y = (np.random.rand(N) < p).astype(int)
        assert np.isclose(p, sum(y) / N, atol=0.05), f"p={p}, p_sim={sum(x) / N,}"
    else:
        x_bar = np.mean(x)
        xy_bar = np.abs(r) * x_bar + (1 - np.abs(r)) * x_bar**2
        toflip = sum(x == 1) - round(len(x) * xy_bar)
        y = x.copy()
        y[np.random.choice(np.where(x == 0)[0], toflip)] = 1
        y[np.random.choice(np.where(x == 1)[0], toflip)] = 0
        if anticorrelation:
            # Invert y to generate anti-correlated variables.
            y = 1 - y
            assert np.isclose(
                1 - p, sum(y) / N, atol=0.05
            ), f"p_y - expected: {1-p}, actual: {sum(y) / N}"

        else:
            assert np.isclose(
                p, sum(y) / N, atol=0.05
            ), f"p_y - expected: {p}, actual: {sum(y) / N}"

        # c = np.corrcoef(x, y)[0, 1]
        # assert np.isclose(c, r, atol=0.05), f"r={r} != c={c} (for p={p})"

    return x, y


def plot_pid_isolated(pid_full, r, title, filename):
    """Plot PID for inputs correlations >= 0"""
    fig, ax = plt.subplots(ncols=4, figsize=(4, 1.4))

    r = np.array(r)
    mask = r > -0.1
    lw = 2
    for k in pid_full:
        pid_full[k] = np.array(pid_full[k])
    measures = ["syn_norm", "shd_norm", "unq_s1_norm", "unq_s2_norm", "mi"]

    # Plot relative estimates.
    for c, m, ind in zip(COLORS, measures, np.arange(len(measures))):
        ax[0].plot(r[mask], pid_full[m][mask], color=c, label=m, linewidth=lw)
        for i, r_plot in enumerate(R_PLOT):
            ax[i + 1].bar(ind, pid_full[m][np.isclose(r, r_plot)], color=c)
            ax[i + 1].set(title="$\\rho=" + str(r_plot) + "$")
    ax[0].plot(
        r[mask],
        (pid_full["shd"][mask] - pid_full["syn"][mask]),
        "k--",
        linewidth=lw,
        label="shd-syn",
    )
    ax[0].set(
        xlabel="$\\rho$",
        xticks=np.arange(0.0, 1.1, 0.2),
        ylabel="abs PID",
        title=title,
        ylim=[-0.7, 1.7],
    )
    for a in ax[1:]:
        a.set(
            xticks=np.arange(len(measures)),
            xticklabels=[
                cfg.labels_short[f"{m}_sx_cor"]
                for m in ["syn_norm", "shd_norm", "unq1_norm", "unq2_norm", "mi"]
            ],
            ylim=[0, 1.05],
        )
        a.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def _print_dist(d, r, gate_name):
    """Print joint distribution to console"""
    x = PrettyTable(["A0", "A1", "T", "p"], float_format="3.4")
    x.title = f"{gate_name} - r={r:.2f}"
    # x.align["Estimator"] = "l"  # Left align
    for realization, p in d.items():
        if p == 0.0:
            continue
        x.add_row(list(realization) + [p])
    print(x)


def plot_input_correlation(p_0, savepath):
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
            title=f"r={r}",
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


def plot_sanity_checks(simulated_c, r, savepath):
    """Plot simulated versus actual correlation as a sanity check"""
    fig = plt.figure(figsize=(3, 3))
    plt.plot(r, simulated_c, color="k", label="actual")
    plt.plot(r, r, color="grey", linestyle=":", label="expected")
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


def _plot_lines(pid_full, r, ax, title):
    """Plot PID over full range of input correlations"""
    lw = 1

    for k in pid_full:
        pid_full[k] = np.array(pid_full[k])

    measures = ["syn", "shd", "unq_s1", "unq_s2", "mi"]
    styles = ["-", "-", "--", ":", "-"]

    # Plot estimates.
    for c, s, m in zip(COLORS, styles, measures):
        ax[0].plot(r, pid_full[m], color=c, label=m, linewidth=lw, linestyle=s)
        ax[1].plot(
            r, pid_full[f"{m}_norm"], color=c, label=m, linewidth=lw, linestyle=s
        )
    ax[0].set(
        xlabel="$\\rho$",
        xticks=[r[0], r[(len(r) - 1) // 2], r[-1]],
        ylabel="PID",
        title=title,
        # ylim=[-0.7, 1.7],
    )
    ax[1].set(
        xlabel="$\\rho$",
        xticks=[r[0], r[(len(r) - 1) // 2], r[-1]],
        ylabel="PID\I",
        title=title,
        # ylim=[-0.7, 1.7],
    )

    # Plot 'reward', i.e., no. ones in output.
    ax[2].plot(r, pid_full["reward"], color="k", linewidth=lw)
    if title == "AND":
        ax[2].plot(r, pid_full["reward_or"], "k--", linewidth=lw)
    ax[2].set(xlabel="$\\rho$", ylabel="$R$")


def main(p_0, estimator, seed):
    """Run estimation"""
    initialize_logger(log_name="logic_gate_pid_comparison")
    savepath = Path("./lbf_logic_gate_pid_comparison/")
    savepath_local = savepath.joinpath("local_sx_pid")
    savepath.mkdir(parents=True, exist_ok=True)
    savepath_local.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # Rdn and Unq1 as used in dit:
    # gates = ["and", "xor", "unq1", "copy", "or", "noisy_xor", "stag", "prisoner"]
    # gates = ["and", "xor", "unq1", "copy"]
    gates = ["and", "xor", "unq1"]
    pid_atoms = ["syn", "shd", "unq_s1", "unq_s2"]

    dist_all = []
    corr_coefficients = [float(i) for i in np.arange(-1.0, 1.1, 0.1)]

    fig, ax = plt.subplots(nrows=3, ncols=len(gates), figsize=(1.2 * len(gates), 4))

    for g, gate_name in enumerate(gates):
        pid_full = {
            k: []
            for k in [
                "mi",
                "mi_norm",
                "coinfo",
                "coinfo_norm",
                "p_joint",
                "corr",
                "reward",
            ]
        }
        for atom in pid_atoms:
            pid_full[atom] = []
            pid_full[f"{atom}_norm"] = []

        for r in corr_coefficients:
            dist, c, reward = correlated_gates(p_0, r, gate_name)
            pid = estimate_pid_from_dist(estimator, dist)
            for atom in pid_atoms:
                pid_full[atom].append(pid[atom])
                pid_full[f"{atom}_norm"].append(pid[f"{atom}_norm"])
            pid_full["coinfo_norm"].append(pid[cfg.col_coinfo_norm])
            pid_full["coinfo"].append(pid[cfg.col_coinfo])
            if gate_name in ["xor", "prisoner"]:
                pid_full["p_joint"].append(dist[(0, 1, 1)])
            else:
                pid_full["p_joint"].append(dist[(1, 1, 1)])
            pid_full["mi"].append(pid[cfg.col_mi])
            pid_full["mi_norm"].append(
                pid[cfg.col_mi]
            )  # add for more convenient plotting
            pid_full["corr"].append(c)
            pid_full["reward"].append(reward)
            dist_all.append(dist)

            if np.any([np.isclose(r, r_plot) for r_plot in R_PLOT]):
                _print_dist(dist, r, gate_name)
                if estimator in ["sx", "sx_cor"]:
                    logging.info("Saving local PID to %s", savepath)
                    plot_local_pid_stats(
                        [calculate_weighted_local_pid(pid["local"])],
                        savepath_local,
                        f"local_pid_{gate_name}_r_" + str(np.around(r, decimals=2)),
                        weighted=True,
                    )
                    if gate_name == "and":
                        dist, c, reward = correlated_gates(
                            p_0, r, gate_name="or", N=10000, eps=0.1
                        )
                        pid = estimate_pid_from_dist(estimator, dist)
                        plot_local_pid_stats(
                            [calculate_weighted_local_pid(pid["local"])],
                            savepath_local,
                            "local_pid_or_r_" + str(np.around(r, decimals=2)),
                            weighted=True,
                        )

        if "and" in gates:
            pid_full["reward_or"] = []
            for r in corr_coefficients:
                _, _, reward = correlated_gates(p_0, r, gate_name="or")
                pid_full["reward_or"].append(reward)

        _plot_lines(pid_full, r=corr_coefficients, ax=ax[:, g], title=gate_name.upper())
        plot_pid_isolated(
            pid_full,
            r=corr_coefficients,
            title=gate_name.upper(),
            filename=savepath.joinpath(
                f"corr_gate_comparison_p_{p_0}_{estimator}_{gate_name.upper()}.{FIGTYPE}"
            ),
        )

    ax[0, 0].legend(loc="upper left", ncol=2)
    ax[1, 0].legend(loc="lower left", ncol=2)
    unify_axis_ylim(ax[:-1, :])
    unify_axis_ylim(ax[-1, :])
    ax[0, -1].set(ylim=[ax[0, 0].get_ylim()[0], ax[0, -1].get_ylim()[1] * 1.1])
    ax[-1, -1].set(ylim=[0, ax[-1, -1].get_ylim()[1] * 1.1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.6, right=0.98)
    logging.info("Saving figures to %s", savepath)
    fig.savefig(
        savepath.joinpath(f"corr_gate_comparison_p_{p_0}_{estimator}.{FIGTYPE}")
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
    parser.add_argument(
        "--estimator",
        "-e",
        type=str,
        default="sx_cor",
        help="PID estimator to use",
    )
    parser.add_argument("--seed", "-s", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--prob", "-p", type=float, default=0.5, help="Probability of taking action 1"
    )
    args = parser.parse_args()
    main(args.prob, args.estimator, args.seed)
