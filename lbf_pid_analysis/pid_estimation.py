#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wrappers around PID estimators to estimate PID for LBF game data.
#
# Copyright (c) Honda Research Institute Europe GmbH
# Copyright (c) Abdullah Makkeh
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

import numpy as np
import pandas as pd

from idtxl.estimators_multivariate_pid import (
    SxPID,
)
import idtxl.lattices as lt
from idtxl import pid_goettingen
import dit
from syndisc.pid import PID_SD

import config as cfg


def estimate_joint_distribution(source1, source2, target):
    """Estimate joint distribution from sources and target time series.

    Parameters
    ----------
    source1 : numpy.ndarray
        Realizations of source 1
    source2 : numpy.ndarray
        Realizations of source 2
    target : numpy.ndarray
        Realizations of the target

    Returns
    -------
    dit.Distribution
        Joint distribution as dit class
    """
    for var in [source1, source2, target]:
        if not np.issubdtype(var.dtype, np.integer):
            raise TypeError("Every variable must be of type integer")
    triplets, counts = np.unique(
        np.vstack((source1, source2, target)).T, axis=0, return_counts=True
    )
    outcomes = [f"{t[0]}{t[1]}{t[2]}" for t in triplets]
    return dit.Distribution(outcomes, counts / len(source1))


def estimate_i_ccs_pid_dit(source1, source2, target, verbose=False):
    """Estimate I_CCS PID by Robin Ince using the dit estimator.

    Parameters
    ----------
    source1 : numpy.ndarray
        Realizations of first source
    source2 : numpy.ndarray
        Realizations of second source
    target : numpy.ndarray
        Realizations of target
    verbose : bool, optional
        Whether to print intermediate steps to console, by default False

    Returns
    -------
    dict
        PID results
    """
    d = estimate_joint_distribution(source1, source2, target)
    if verbose:
        print("dit distribution:\n", d)
    pid = dit.pid.PID_CCS(d)
    logging.info(
        "Estimated synergy (I_ccs): %.5f",
        pid.get_pi(cfg.i_ccs_lattice["syn"]),
    )
    result = {
        cfg.col_unq_s1: pid.get_pi(cfg.i_ccs_lattice["unq1"]),
        cfg.col_unq_s2: pid.get_pi(cfg.i_ccs_lattice["unq2"]),
        cfg.col_shd: pid.get_pi(cfg.i_ccs_lattice["shd"]),
        cfg.col_syn: pid.get_pi(cfg.i_ccs_lattice["syn"]),
    }
    result[cfg.col_mi] = (
        result[cfg.col_unq_s1]
        + result[cfg.col_unq_s2]
        + result[cfg.col_shd]
        + result[cfg.col_syn]
    )
    result[cfg.col_syn_norm] = (
        result[cfg.col_syn] / result[cfg.col_mi]
        if result[cfg.col_mi] != 0
        else 0
    )
    return result


def estimate_sx_pid_from_dist(pdf, correction=False, num_source_vars=2):
    # Read lattices from a file
    # Stored as {
    #             n -> [{alpha -> children}, (alpha_1,...) ]
    #           }
    # children is a list of tuples
    lattices = lt.lattices
    retval_ptw, retval_avg = pid_goettingen.pid(
        num_source_vars,
        pdf_orig=pdf,
        chld=lattices[num_source_vars][0],
        achain=lattices[num_source_vars][1],
        printing=False,
    )
    pid_sx = {
        cfg.col_unq_s1: retval_avg[cfg.sx_pid_lattice["unq1"]][2],
        cfg.col_unq_s2: retval_avg[cfg.sx_pid_lattice["unq2"]][2],
        cfg.col_shd: retval_avg[cfg.sx_pid_lattice["shd"]][2],
        cfg.col_syn: retval_avg[cfg.sx_pid_lattice["syn"]][2],
        "local": retval_ptw,
    }
    if correction:
        pid_sx = correct_sx_pid(pid_sx)
    return _prepare_sx_pid_output(pid_sx, correction)


def correct_sx_pid(pid_sx):
    if pid_sx[cfg.col_unq_s1] < 0 or pid_sx[cfg.col_unq_s2] < 0:
        correct_by = np.abs(
            np.min([pid_sx[cfg.col_unq_s1], pid_sx[cfg.col_unq_s2]])
        )
        pid_sx[cfg.col_unq_s1] += correct_by
        pid_sx[cfg.col_unq_s2] += correct_by
        pid_sx[cfg.col_syn] -= correct_by
        pid_sx[cfg.col_shd] -= correct_by
        logging.info(
            "Correcting by %.4f for negative unique in at least one source",
            correct_by,
        )
    elif pid_sx[cfg.col_shd] < 0:
        correct_by = np.abs(pid_sx[cfg.col_shd])
        pid_sx[cfg.col_unq_s1] -= correct_by
        pid_sx[cfg.col_unq_s2] -= correct_by
        pid_sx[cfg.col_syn] += correct_by
        pid_sx[cfg.col_shd] += correct_by
        logging.info(
            "Correcting by %.4f for negative shared information", correct_by
        )
    else:
        logging.info("No correction needed")
    return pid_sx


def _prepare_sx_pid_output(pid_sx, correction):
    for atom in (
        cfg.col_unq_s1,
        cfg.col_unq_s2,
        cfg.col_syn,
        cfg.col_shd,
    ):
        if np.isclose(pid_sx[atom], 0):
            pid_sx[atom] = 0

    pid_sx[cfg.col_mi] = (
        pid_sx[cfg.col_unq_s1]
        + pid_sx[cfg.col_unq_s2]
        + pid_sx[cfg.col_syn]
        + pid_sx[cfg.col_shd]
    )
    pid_sx[cfg.col_syn_norm] = (
        pid_sx[cfg.col_syn] / pid_sx[cfg.col_mi]
        if pid_sx[cfg.col_mi] != 0
        else 0
    )
    if correction:
        logging.info("Estimated synergy (I_sx_cor): %.5f", pid_sx[cfg.col_syn])
    else:
        logging.info("Estimated synergy (I_sx): %.5f", pid_sx[cfg.col_syn])
    return pid_sx


def estimate_sx_pid(source1, source2, target, correction=False):
    """Estimate SxPID by Abed Makkeh and Michael Wibral.

    Parameters
    ----------
    source1 : numpy.ndarray
        Realizations of first source
    source2 : numpy.ndarray
        Realizations of second source
    target : numpy.ndarray
        Realizations of target
    correction : bool, optional
        Whether to correct violations of the consistency equations on a global
        level, by default False

    Returns
    -------
    dict
        PID results
    """
    sx_pid_est = SxPID({})
    pid_sx_alpha = sx_pid_est.estimate([source1, source2], target)

    triplets = np.vstack((source1, source2, target)).T
    triplet, counts = np.unique(triplets, axis=0, return_counts=True)
    n_samples = len(source1)
    d = []
    for k in pid_sx_alpha["ptw"]:
        p = pid_sx_alpha["ptw"][k]
        lmi = (
            p[((1,),)][2] + p[((2,),)][2] + p[((1,), (2,))][2] + p[((1, 2),)][2]
        )

        ind = np.where((triplet == (k[0], k[1], k[2])).all(axis=1))[0][0]
        d.append(
            {
                cfg.col_s1: k[0],
                cfg.col_s2: k[1],
                cfg.col_t: k[2],
                cfg.col_unq_s1: p[((1,),)][2],
                cfg.col_unq_s2: p[((2,),)][2],
                cfg.col_shd: p[((1,), (2,))][2],
                cfg.col_syn: p[((1, 2),)][2],
                cfg.col_count: counts[ind],
                cfg.col_joint_prob: counts[ind] / n_samples,
                cfg.col_lmi_s1_s2_t: lmi,
                cfg.col_lmi_s1_t: p[((1,),)][2] + p[((1,), (2,))][2],
                cfg.col_lmi_s2_t: p[((2,),)][2] + p[((1,), (2,))][2],
            }
        )
    pid_sx_local = pd.DataFrame(d).sort_values(
        by=[cfg.col_t, cfg.col_s1, cfg.col_s2]
    )
    pid_sx = {
        cfg.col_unq_s1: pid_sx_alpha["avg"][cfg.sx_pid_lattice["unq1"]][2],
        cfg.col_unq_s2: pid_sx_alpha["avg"][cfg.sx_pid_lattice["unq2"]][2],
        cfg.col_shd: pid_sx_alpha["avg"][cfg.sx_pid_lattice["shd"]][2],
        cfg.col_syn: pid_sx_alpha["avg"][cfg.sx_pid_lattice["syn"]][2],
        "local": pid_sx_local,
    }
    assert pid_sx_local[cfg.col_count].sum() == n_samples
    assert np.isclose(
        (pid_sx_local[cfg.col_count] * pid_sx_local[cfg.col_syn]).sum()
        / pid_sx_local[cfg.col_count].sum(),
        pid_sx[cfg.col_syn],
    ), "Average of local synergy is not equal to total synergy"

    if correction:
        pid_sx = correct_sx_pid(pid_sx)
    return _prepare_sx_pid_output(pid_sx, correction)


def estimate_pid_syndisc(s1, s2, t, verbose=False):
    """Estimate PID using the synergy-based synergistic disclosure estimator

    Parameters
    ----------
    s1 : numpy.ndarray
        Realizations source 1
    s2 : numpy.ndarray
        Realizations source 2
    t : numpy.ndarray
        Realizations target
    verbose : bool, optional
        Whether to print progress to console, by default False

    Returns
    -------
    dict
        PID estimates
    """
    d = estimate_joint_distribution(s1, s2, t)
    if verbose:
        print("dit distribution:\n", d)

    def _get_results(pid):
        return {
            cfg.col_unq_s1: pid.get_pi(cfg.syndisc_lattice["unq1"]),
            cfg.col_unq_s2: pid.get_pi(cfg.syndisc_lattice["unq2"]),
            cfg.col_shd: pid.get_pi(cfg.syndisc_lattice["shd"]),
            cfg.col_syn: pid.get_pi(cfg.syndisc_lattice["syn"]),
            cfg.col_mi: (
                pid.get_pi(cfg.syndisc_lattice["unq1"])
                + pid.get_pi(cfg.syndisc_lattice["unq2"])
                + pid.get_pi(cfg.syndisc_lattice["shd"])
                + pid.get_pi(cfg.syndisc_lattice["syn"])
            ),
        }

    if verbose:
        print("Estimate synergistic disclosure")
    res = PID_SD(d)
    if verbose:
        print("done")
    pid = _get_results(res)
    pid[cfg.col_mi] = (
        pid[cfg.col_unq_s1]
        + pid[cfg.col_unq_s2]
        + pid[cfg.col_syn]
        + pid[cfg.col_shd]
    )
    pid[cfg.col_syn_norm] = (
        pid[cfg.col_syn] / pid[cfg.col_mi] if pid[cfg.col_mi] != 0 else 0
    )
    logging.info("Estimated synergy (syndisc): %.5f", pid[cfg.col_syn])
    return pid
