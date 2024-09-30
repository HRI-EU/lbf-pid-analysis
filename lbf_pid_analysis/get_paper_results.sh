#! /bin/bash
#
# Run experiments without distractors.
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
set -e -u -o pipefail

# ------------------------------------------------------------- dist = 0.0
LOADPATH="../../lbf_experiments/shared_goal_dist_0_0_v13/"

for TARGET in any_food n_collections_agent_0
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 3 4 5 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 6 7 8 9 10 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 11 12 13 14 15 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 16 17 18 19 20 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 21 22 23 24 25 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 26 27 28 29 30 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 31 32 33 34 35 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 36 37 38 39 40 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 41 42 43 44 45 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 46 47 48 49 50 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 51 52 53 54 55 --target "${TARGET}" --path "${LOADPATH}"
done

# SxPID without and with correction
echo "${LOADPATH}"
python plot_measure_by_c.py -m mi_sx shd_norm_sx syn_norm_sx unq1_norm_sx unq2_norm_sx -p "${LOADPATH}" -f 55
python plot_measure_by_c.py -m mi_sx_cor syn_norm_sx_cor shd_norm_sx_cor unq1_norm_sx_cor unq2_norm_sx_cor -p "${LOADPATH}" -f 55

# Local PID plots for subset of settings and trials
TARGET="any_food"
echo "${TARGET}"
python estimate_local_pid.py -f 55 --target "${TARGET}" --path "${LOADPATH}"
for FOLDER in 1 2 3 4 5 26 27 28 29 30 41 42 43 44 45 51 52 53 54 55
do
    for TRIAL in {0..5}
    do
        python plot_local_sx_pid_per_trial.py --trials 3 -t "${TARGET}" -f "${FOLDER}" --path "${LOADPATH}"
    done
done

# ------------------------------------------------------------- asymmetric
LOADPATH="../../lbf_experiments/asymmetric_d_0_0_v13/"

for TARGET in any_food n_collections_agent_0 n_collections_agent_1
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 3 4 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 5 6 7 8 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 9 10 11 12 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 13 14 15 16 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 17 18 19 20 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 21 22 23 24 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 25 26 27 28 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 29 30 31 32 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 33 34 35 36 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 37 38 39 40 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 41 42 43 44 --target "${TARGET}" --path "${LOADPATH}"
done

python plot_pid_asymmetric_cond.py --path "${LOADPATH}"
