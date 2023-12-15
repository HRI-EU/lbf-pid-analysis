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
import pytest

import numpy as np

sys.path.insert(0, "../lbf_pid_analysis")
from mi_estimation import estimate_h_discrete
from analyze_pid_per_trial import estimate_joint_mutual_information


def test_entropy_estimation():
    """Test estimation of a variable's entropy"""
    np.random.seed(1)
    entropy = estimate_h_discrete(np.ones(100))
    assert entropy == 0
    entropy = estimate_h_discrete(np.random.randint(0, 2, size=100))
    assert np.isclose(entropy, 1, rtol=0.02)
    with pytest.raises(TypeError):
        estimate_h_discrete(np.random.randint(0, 2, size=100).astype(float))
    with pytest.raises(ValueError):
        estimate_h_discrete(np.random.randint(0, 2, size=100) * -1)


def test_mutual_information_estimation():
    """Test estimation of mutual information from two variables"""
    np.random.seed(1)
    N = 100
    source1 = np.random.randint(0, 2, size=N, dtype=int)
    source2 = np.random.randint(0, 2, size=N, dtype=int)
    target = np.logical_xor(source1, source2).astype(int)
    settings = {"analysis": {"nperm": 21}}
    with pytest.raises(TypeError):
        estimate_joint_mutual_information(
            source1, source2, target.astype(float), n_perm=100, seed=1
        )
    with pytest.raises(ValueError):
        estimate_joint_mutual_information(
            source1, source2, target * -1, n_perm=100, seed=1
        )

    target_entropy, mi, sign_mi, p = estimate_joint_mutual_information(
        source1, source2, target, n_perm=100, seed=1
    )
    assert np.isclose(target_entropy, 1, rtol=0.01)
    assert np.isclose(mi, 1, rtol=0.01)
    assert np.isclose(target_entropy, mi, rtol=0.01)
    assert sign_mi
    assert p < 0.05


if __name__ == "__main__":
    test_entropy_estimation()
    test_mutual_information_estimation()
