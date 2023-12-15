#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run tests for PID analysis of LBF experiments.
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
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "../lbf_pid_analysis")
from pid_estimation import (  # pylint: disable=E0401,C0413,C0411
    estimate_joint_distribution,
    estimate_sx_pid,
    estimate_i_ccs_pid_dit,
    estimate_pid_syndisc,
)

COL_SYNERGY = "syn"
COL_SHARED = "shd"


def test_estimate_joint_distribution():
    """Generate a joint distribution in the dit format from time series"""
    N = 100

    with pytest.raises(TypeError):
        estimate_joint_distribution(
            np.ones(N, dtype=float), np.ones(N), np.ones(N)
        )

    source1 = np.ones(N, dtype=int)
    source2 = np.ones(N, dtype=int)
    target1 = np.ones(N, dtype=int)

    distribution = estimate_joint_distribution(source1, source2, target1)
    print(distribution)
    for alphabet in distribution.alphabet:
        assert alphabet == ("1",)
    for outcome in distribution.outcomes:
        assert outcome == "111"
    assert distribution.pmf == 1.0

    target2 = np.hstack(
        (np.ones(N // 2, dtype=int), np.zeros(N // 2, dtype=int))
    )
    distribution = estimate_joint_distribution(source1, source2, target2)
    print(distribution)
    for alphabet in distribution.alphabet[:2]:
        assert alphabet == ("1",)
    for outcome, expected_outcome in zip(distribution.outcomes, ["110", "111"]):
        assert outcome == expected_outcome
    for prob in distribution.pmf:
        assert prob == 0.5


def test_pid_estimators():
    """Test estimation of PID using different estimators"""
    np.random.seed(1)
    N = 10000
    source1 = np.random.randint(0, 2, size=N, dtype=int)
    source2 = np.random.randint(0, 2, size=N, dtype=int)

    target_xor = np.logical_xor(source1, source2).astype(int)
    pid = estimate_sx_pid(source1, source2, target_xor, correction=True)
    print("SxPID XOR:\n", pid)
    assert np.isclose(pid[COL_SYNERGY], 1, rtol=0.015)
    assert np.isclose(pid["unq_s1"], 0, atol=0.0001)
    assert np.isclose(pid["unq_s2"], 0, atol=0.0001)
    assert np.isclose(pid["shd"], 0, atol=0.0001)
    pid = estimate_i_ccs_pid_dit(source1, source2, target_xor)
    assert np.isclose(pid[COL_SYNERGY], 1, rtol=0.015)
    pid = estimate_pid_syndisc(source1, source2, target_xor)
    assert np.isclose(pid[COL_SYNERGY], 1, rtol=0.2)

    target_and = np.logical_and(source1, source2).astype(int)
    pid = estimate_sx_pid(source1, source2, target_and, correction=True)
    assert np.isclose(pid[COL_SYNERGY], 0.311, rtol=0.015)
    pid = estimate_i_ccs_pid_dit(source1, source2, target_and)
    assert np.isclose(pid[COL_SYNERGY], 0.311, rtol=0.1)
    pid = estimate_pid_syndisc(source1, source2, target_and)
    assert np.isclose(pid[COL_SYNERGY], 0.311, rtol=0.015)

    with pytest.raises(TypeError):
        estimate_sx_pid(source1, source2, np.arange(10, dtype=float))
    with pytest.raises(TypeError):
        estimate_i_ccs_pid_dit(source1, source2, np.arange(10, dtype=float))
    with pytest.raises(TypeError):
        estimate_pid_syndisc(source1, source2, np.arange(10, dtype=float))


if __name__ == "__main__":
    test_estimate_joint_distribution()
    test_pid_estimators()
