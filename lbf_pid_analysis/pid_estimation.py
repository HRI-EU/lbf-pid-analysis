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


def _get_iccs_results(pid):
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
    return result


def estimate_pid(measure, source1, source2, target):
    """Estimate PID using the specified measure and estimator.

    Estimate PID atoms as well as normalized atoms and cooperation measures
    co-information and the ratio of shared and synergistic information (from
    absolute and normalized atoms).

    Parameters
    ----------
    measure : str
        Measure to use ('sx', 'sx_cor', 'iccs', 'broja', 'syndisc')
    source1 : numpy.ndarray
        Realizations source 1
    source2 : numpy.ndarray
        Realizations source 2
    target : numpy.ndarray
        Realizations target

    Returns
    -------
    dict
        PID estimates

    Raises
    ------
    RuntimeError
        If requested measure is unknown
    """
    if measure == "sx":
        pid = estimate_sx_pid(source1, source2, target, correction=False, verbose=False)
    elif measure == "sx_cor":
        pid = estimate_sx_pid(source1, source2, target, correction=True, verbose=False)
    elif measure == "iccs":
        pid = estimate_i_ccs_pid_dit(source1, source2, target, verbose=False)
    elif measure == "broja":
        pid = estimate_broja_pid_dit(source1, source2, target, verbose=False)
    elif measure == "syndisc":
        pid = estimate_pid_syndisc(source1, source2, target, verbose=False)
    else:
        raise RuntimeError("Unknown measure %s", measure, verbose=False)
    pid = _calculate_normalized_atoms(pid)
    return _calculate_coop_measures(pid)


def estimate_pid_from_dist(measure, dist):
    """Estimate PID from a joint dist using the specified measure and estimator.

    Estimate PID atoms as well as normalized atoms and cooperation measures
    co-information and the ratio of shared and synergistic information (from
    absolute and normalized atoms).

    Parameters
    ----------
    measure : str
        Measure to use ('sx', 'sx_cor', 'iccs', 'broja', 'syndisc')
    dist : dict
        Joint distribution to estimate from

    Returns
    -------
    dict
        PID estimates

    Raises
    ------
    RuntimeError
        If requested measure is unknown
    """
    if measure == "sx":
        pid = estimate_sx_pid_from_dist(dist, correction=False, verbose=False)
    elif measure == "sx_cor":
        pid = estimate_sx_pid_from_dist(dist, correction=True, verbose=False)
    elif measure == "iccs":
        pid = estimate_i_ccs_pid_dit_from_dist(dist, verbose=False)
    elif measure == "broja":
        pid = estimate_broja_pid_from_dist(dist, verbose=False)
    elif measure == "syndisc":
        pid = estimate_pid_syndisc_from_dist(dist, verbose=False)
    else:
        raise RuntimeError("Unknown measure %s", measure, verbose=False)
    pid = _calculate_normalized_atoms(pid)
    return _calculate_coop_measures(pid)


def _calculate_normalized_atoms(pid):
    # Normalize PID atoms by the joint mutual information.
    for atom in ["syn", "shd", "unq_s1", "unq_s2"]:
        if pid["mi"] != 0:
            pid[f"{atom}_norm"] = pid[atom] / pid["mi"]
        else:
            pid[f"{atom}_norm"] = 0
    return pid


def _calculate_coop_measures(pid):
    # Calculate measures of cooperation (coInfo and ratio shd/syn)
    pid["coinfo"] = np.array(pid[cfg.col_shd]) - np.array(pid[cfg.col_syn])
    pid["coinfo_norm"] = np.array(pid[cfg.col_shd_norm]) - np.array(
        pid[cfg.col_syn_norm]
    )
    pid["coop_ratio"] = np.divide(
        np.array(pid[cfg.col_shd], dtype=float),
        np.array(pid[cfg.col_syn], dtype=float),
        out=np.zeros_like(np.array(pid[cfg.col_shd]), dtype=float),
        where=np.array(pid[cfg.col_syn]) != 0,
    )
    pid["coop_ratio_norm"] = np.divide(
        np.array(pid["shd_norm"], dtype=float),
        np.array(pid["syn_norm"], dtype=float),
        out=np.zeros_like(np.array(pid["shd_norm"]), dtype=float),
        where=np.array(pid["syn_norm"]) != 0,
    )
    return pid


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
    return _get_iccs_results(pid)


def estimate_i_ccs_pid_dit_from_dist(dist, verbose=False):
    """Estimate I_CCS PID by Robin Ince using the dit estimator.

    Parameters
    ----------
    dist : dict
        Joint distribution
    verbose : bool, optional
        Whether to print intermediate steps to console, by default False

    Returns
    -------
    dict
        PID results
    """
    pid = dit.pid.PID_CCS(dit.Distribution([k for k in dist], [dist[k] for k in dist]))
    return _get_iccs_results(pid)


def _prepare_local_pid(ptw_pid, triplets, abscounts=None, relcounts=None):
    if abscounts is not None:
        n_samples = sum(abscounts)
        if relcounts is not None:
            raise RuntimeError("Provide either abscounts or relcounts")
        relcounts = np.array(abscounts) / n_samples
    else:
        abscounts = np.zeros(len(triplets), dtype=int)
        if relcounts is None:
            raise RuntimeError("Provide either abscounts or relcounts")
    d = []
    for k in ptw_pid:
        p = ptw_pid[k]
        lmi = p[((1,),)][2] + p[((2,),)][2] + p[((1,), (2,))][2] + p[((1, 2),)][2]

        ind = np.where((triplets == (k[0], k[1], k[2])).all(axis=1))[0][0]
        d.append(
            {
                cfg.col_s1: k[0],
                cfg.col_s2: k[1],
                cfg.col_t: k[2],
                cfg.col_unq_s1: p[((1,),)][2],
                cfg.col_unq_s2: p[((2,),)][2],
                cfg.col_shd: p[((1,), (2,))][2],
                cfg.col_syn: p[((1, 2),)][2],
                cfg.col_count: abscounts[ind],
                cfg.col_joint_prob: relcounts[ind],
                cfg.col_lmi_s1_s2_t: lmi,
                cfg.col_lmi_s1_t: p[((1,),)][2] + p[((1,), (2,))][2],
                cfg.col_lmi_s2_t: p[((2,),)][2] + p[((1,), (2,))][2],
            }
        )
    pid_sx_local = pd.DataFrame(d).sort_values(by=[cfg.col_t, cfg.col_s1, cfg.col_s2])
    return pid_sx_local


def estimate_sx_pid(source1, source2, target, correction=False, verbose=True):
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
    verbose : bool, optional
        Whether to print logging output, by default True

    Returns
    -------
    dict
        PID results
    """
    sx_pid_est = SxPID({})
    pid_sx_alpha = sx_pid_est.estimate([source1, source2], target)
    pid_sx = {
        cfg.col_unq_s1: pid_sx_alpha["avg"][cfg.sx_pid_lattice["unq1"]][2],
        cfg.col_unq_s2: pid_sx_alpha["avg"][cfg.sx_pid_lattice["unq2"]][2],
        cfg.col_shd: pid_sx_alpha["avg"][cfg.sx_pid_lattice["shd"]][2],
        cfg.col_syn: pid_sx_alpha["avg"][cfg.sx_pid_lattice["syn"]][2],
    }
    triplets, counts = np.unique(
        np.vstack((source1, source2, target)).T, axis=0, return_counts=True
    )
    pid_sx_local = _prepare_local_pid(
        pid_sx_alpha["ptw"], triplets=triplets, abscounts=counts
    )
    assert np.isclose(
        (
            pid_sx_local[cfg.col_count]  # pylint: disable=E1136
            * pid_sx_local[cfg.col_syn]  # pylint: disable=E1136
        ).sum()
        / pid_sx_local[cfg.col_count].sum(),  # pylint: disable=E1136
        pid_sx[cfg.col_syn],
    ), "Average of local synergy is not equal to total synergy"
    pid_sx["local"] = pid_sx_local
    if correction:
        pid_sx = correct_sx_pid(pid_sx, verbose)
    return _prepare_sx_pid_output(pid_sx, correction, verbose)


def estimate_sx_pid_from_dist(pdf, correction=False, num_source_vars=2, verbose=True):
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

    pid_sx_local = _prepare_local_pid(
        retval_ptw,
        triplets=np.array([np.array(k) for k in pdf.keys()]),
        relcounts=list(pdf.values()),
    )
    pid_sx = {
        cfg.col_unq_s1: retval_avg[cfg.sx_pid_lattice["unq1"]][2],
        cfg.col_unq_s2: retval_avg[cfg.sx_pid_lattice["unq2"]][2],
        cfg.col_shd: retval_avg[cfg.sx_pid_lattice["shd"]][2],
        cfg.col_syn: retval_avg[cfg.sx_pid_lattice["syn"]][2],
        "local": pid_sx_local,
    }
    if correction:
        pid_sx = correct_sx_pid(pid_sx, verbose)
    return _prepare_sx_pid_output(pid_sx, correction, verbose)


def correct_sx_pid(pid_sx, verbose):
    if pid_sx[cfg.col_unq_s1] < 0 or pid_sx[cfg.col_unq_s2] < 0:
        correct_by = np.abs(np.min([pid_sx[cfg.col_unq_s1], pid_sx[cfg.col_unq_s2]]))
        pid_sx[cfg.col_unq_s1] += correct_by
        pid_sx[cfg.col_unq_s2] += correct_by
        pid_sx[cfg.col_syn] -= correct_by
        pid_sx[cfg.col_shd] -= correct_by
        if verbose:
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
        if verbose:
            logging.info(
                "Correcting by %.4f for negative shared information", correct_by
            )
    else:
        if verbose:
            logging.info("No correction needed")
    return pid_sx


def _prepare_sx_pid_output(pid_sx, correction, verbose=True):
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
    if verbose:
        if correction:
            logging.info("Estimated synergy (I_sx_cor): %.5f", pid_sx[cfg.col_syn])
        else:
            logging.info("Estimated synergy (I_sx): %.5f", pid_sx[cfg.col_syn])
    return pid_sx


def _get_syndisc_results(pid, verbose=False):
    pid = {
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
    pid[cfg.col_mi] = (
        pid[cfg.col_unq_s1] + pid[cfg.col_unq_s2] + pid[cfg.col_syn] + pid[cfg.col_shd]
    )
    if verbose:
        logging.info("Estimated synergy (syndisc): %.5f", pid[cfg.col_syn])
    pid["local"] = []
    return pid


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

    if verbose:
        print("Estimate synergistic disclosure")
    res = PID_SD(d)
    if verbose:
        print("done")
    return _get_syndisc_results(res, verbose)


def estimate_pid_syndisc_from_dist(dist, verbose=False):
    if verbose:
        print("Estimate synergistic disclosure")
    res = PID_SD(dit.Distribution([k for k in dist], [dist[k] for k in dist]))
    if verbose:
        print("done")
    return _get_syndisc_results(res, verbose)


def estimate_broja_pid_dit(source1, source2, target, verbose=False):
    """Estimate BROJA PID by Bertschinger et al. using the dit estimator.

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
    pid = dit.pid.PID_BROJA(d)
    return _get_iccs_results(pid)


def estimate_broja_pid_from_dist(dist, verbose=False):
    """Estimate BROJA PID by Bertschinger et al. using the dit estimator.

    Parameters
    ----------
    dist : dict
        Joint distribution
    verbose : bool, optional
        Whether to print intermediate steps to console, by default False

    Returns
    -------
    dict
        PID results
    """
    pid = dit.pid.PID_BROJA(
        dit.Distribution([k for k in dist], [dist[k] for k in dist])
    )
    return _get_iccs_results(pid)
