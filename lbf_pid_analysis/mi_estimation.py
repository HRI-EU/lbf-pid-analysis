#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Discrete entropy estimator using the JIDT Java toolbox.
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

import numpy as np
import jpype as jp

import idtxl.idtxl_exceptions as ex
import idtxl.idtxl_utils as utils
from idtxl.estimators_jidt import (
    JidtDiscrete,
    JidtDiscreteMI,
)
from idtxl.stats import _find_pvalue


def _check_data_for_mi_estimation(x1, x2, discretize_method, n_bins=None):
    if x1.shape[0] != x2.shape[0]:
        raise RuntimeError("Inputs x and y must be of equal length!")
    if discretize_method is not None:
        settings = {"discretize_method": discretize_method}
        if isinstance(n_bins, (int, float)):
            settings["n_discrete_bins"] = n_bins
        elif isinstance(n_bins, (dict)):
            try:
                settings.update({"alph1": n_bins["x1"], "alph2": n_bins["x2"]})
            except KeyError as err:
                logging.info("If n_bins is a dict, it has to contain keys 'x1', 'x2'.")
                logging.info(err.args)
                raise
        else:
            raise RuntimeError(
                "Input n_bins has to be numerical or dict, got", type(n_bins)
            )
        return settings
    return {}


def estimate_mi_discrete(
    x1,
    x2,
    alpha=0.05,
    n_perm=500,
    n_bins=2,
    discretize_method="equal",
    seed=None,
):
    """Estimate mutual information and perform permutation test of estimate.

    Estimate mutual information using the Kraskov estimator. Perform a
    permutation test to assess whether estimate is significantly bigger than
    zero.

    References
    ----------
    - Kraskov, Stögbauer, & Grassberger (2004). Estimating mutual information.
      Phys Rev E, 69(6), 16. https://doi.org/10.1103/PhysRevE.69.066138

    Parameters
    ----------
    x1 : numpy array
        Realizations of variable 1
    x2 : numpy array
        Realizations of variable 2
    alpha : float, optional
        Critical alpha level, by default 0,05
    n_perm : int | None, optional
        Number of permutations, if None no test is performed, by default 500
    n_bins: int | dict, optional
        Number of bins for discretization, None for already discretized
        variables, dict with individual numbers of bins, or int to use same no.
        bins for all variables, by default 2
    discretize_method : str
        Discretization method, can be 'max_ent' for maximum entropy binning,
        'equal' for equal size bins, and 'none' if no binning is required, by
        default 'none'
    seed : int, optional
        Random seed

    Returns
    -------
    significance : bool
        If p-value is below critical alpha level
    p_value : float
        The test's p-value
    cutoff : float
        The test's significance cutoff given the p-value
    """
    np.random.seed(seed)

    settings = _check_data_for_mi_estimation(x1, x2, discretize_method, n_bins)
    est_mi_discrete = JidtDiscreteMI(settings)
    mi = est_mi_discrete.estimate(x1, x2)

    permutation_distribution = np.zeros(n_perm)
    x2_surrogates = x2.copy()
    for permutation in range(n_perm):
        permutation_distribution[permutation] = est_mi_discrete.estimate(
            x1, np.random.permutation(x2_surrogates)
        )
    significance, p_value = _find_pvalue(
        statistic=mi,
        distribution=permutation_distribution,
        alpha=alpha,
        tail="one_bigger",
    )
    logging.debug(  # pylint: disable=W1201
        "Surrogate distribution: %s" % permutation_distribution  # pylint: disable=C0209
    )
    return mi, significance, p_value


def estimate_h_discrete(target):
    """Estimate target entropy

    Parameters
    ----------
    target : numpy.ndarray
        Realizations of target variable

    Returns
    -------
    float
        Estimated entropy
    """
    settings = {
        "cmi_estimator": "JidtDiscreteCMI",
        "local_values": False,
        "discretize_method": "none",
        "alph": len(np.unique(target)),
        "verbose": False,
    }
    if settings["alph"] == 1:
        logging.info("Constant target, returning H(T) = 0")
        return 0
    est = JidtDiscreteH(settings)
    h = est.estimate(target)
    logging.debug("Estimated target entropy, H(T): %.5f", h)
    return h


class JidtDiscreteH(JidtDiscrete):
    """Calculate entropy with JIDT's implementation for discrete variables.

    Calculate the entropy of a variable using JIDT's discrete Java estimator. See parent
    class for references. Uses jpype to call Java from Python.

    Parameters
    ----------
    settings : dict [optional]
        Estimation parameters. Can contain:

        - debug : bool [optional] - return debug information when calling
            JIDT (default=False)
        - local_values : bool [optional] - return local TE instead of
            average TE (default=False)
        - discretize_method : str [optional] - if and how to discretize
            incoming continuous data, can be 'max_ent' for maximum entropy
            binning, 'equal' for equal size bins, and 'none' if no binning is
            required (default='none')
        - n_discrete_bins : int [optional] - number of discrete bins/
            levels or the base of each dimension of the discrete variables
            (default=2). If set, this parameter overwrites/sets alph
        - alph : int [optional] - alphabet size of discrete variable
            (default=2)

    ex.JidtOutOfMemoryError
        _description_
    """

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        # Set default alphabet sizes. Try to overwrite alphabet sizes with
        # number of bins for discretization if provided, otherwise assume
        # binary variables.
        try:
            settings["alph"] = int(settings["n_discrete_bins"])
        except KeyError:
            pass  # Do nothing and use the default for alph_* set below
        settings.setdefault("alph", int(2))
        super().__init__(settings)

        # Start JAVA virtual machine and create JAVA object. Add JAVA object to
        # instance, the discrete estimator requires the variable dimensions
        # upon instantiation.
        self._start_jvm()
        self.estimator_class = jp.JPackage(
            "infodynamics.measures.discrete"
        ).EntropyCalculatorDiscrete

    def estimate(self, var):
        """Estimate entropy.

        Parameters
        ----------
        var : numpy array
            realizations of variable, either a 2D numpy array where array dimensions
            represent [realizations x variable dimension] or a 1D array representing
            [realizations], array type can be float (requires discretization) or int

        Returns
        -------
        float
            Estimated entropy
        """
        # Check and remember the no. dimensions for each variable before
        # collapsing them into univariate arrays later.
        var = self._ensure_two_dim_input(var)
        var_dim = var.shape[1]

        # Discretize if requested.
        if self.settings["discretize_method"] == "none":
            if not issubclass(var.dtype.type, np.integer):
                raise TypeError(
                    "If no discretization is chosen, data has to be an integer array."
                )
            if var.min() < 0:
                raise ValueError("Minimum of process is smaller than 0.")
            if var.max() >= self.settings["alph"]:
                raise ValueError("Maximum of process is larger than the alphabet size.")
            if self.settings["alph"] < np.unique(var).shape[0]:
                raise RuntimeError(
                    "The process alphabet size does not match the no. unique elements in the process."
                )
        elif self.settings["discretize_method"] == "equal":
            var = utils.discretise(var, self.settings["alph"])
        elif self.settings["discretize_method"] == "max_ent":
            var = utils.discretise_max_ent(var, self.settings["alph"])
        else:
            pass  # don't discretize at all, assume data to be discrete

        # Collapse > 1D arrays into 1D arrays
        var = utils.combine_discrete_dimensions(var, self.settings["alph"])

        # We have a non-trivial conditional, so make a proper conditional MI
        # calculation
        var_base = int(np.power(self.settings["alph"], var_dim))
        try:
            calc = self.estimator_class(var_base)
        except:
            # Only possible exception that can be raised here
            #  (if all bases >= 2) is a Java OutOfMemoryException:
            assert var_base >= 2
            raise ex.JidtOutOfMemoryError(
                (
                    "Cannot instantiate JIDT CMI discrete estimator with alph_base = %d. "
                    "Try re-running increasing Java heap size"
                )
                % var_base
            )
        calc.setDebug(self.settings["debug"])
        calc.initialise()
        # Unfortunately no faster way to pass numpy arrays in than this list
        # conversion
        calc.addObservations(jp.JArray(jp.JInt, 1)(var.tolist()))
        if self.settings["local_values"]:
            return np.array(calc.computeLocal(jp.JArray(jp.JInt, 1)(var.tolist())))
        return float(calc.computeAverageLocalOfObservations())

    def get_analytic_distribution(self, **data):
        raise NotImplementedError("Returning the analytic distribution")
