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

# ------------------------------------------------------ Symmetric agent levels
LOADPATH="../../lbf_experiments/shared_goal_dist_0_0_v13/"

python plot_measure_by_c.py --path "${LOADPATH}" -f 55 -m unq1_norm_sx_cor shd_norm_sx_cor syn_norm_sx_cor  # PID by c and heuristic
python plot_measure_by_c.py --path "${LOADPATH}" -f 55 -m any_food_collected n_cooperative_actions mi_sx # manipulation check

python compare_pid_and_task_success.py --path "${LOADPATH}" -f 55 -c 0.8 --success any_food_collected  # PID profiles for similar task success

python correlate_pid.py --path "${LOADPATH}" -f 55 -x p_joint_action -y shd_norm_sx_cor
python correlate_pid.py --path "${LOADPATH}" -f 55 -x p_indiv1_action -y unq1_norm_sx_cor
python correlate_pid.py --path "${LOADPATH}" -f 55 -x local_shd_joint_action -y mi_sx_cor
python correlate_pid.py --path "${LOADPATH}" -f 55 -x local_unq1_indiv_action -y mi_sx_cor

python plot_local_sx_pid_per_trial.py --path "${LOADPATH}" --trials 3 --folder 2  # Ego, c=0.0
python plot_local_sx_pid_per_trial.py --path "${LOADPATH}" --trials 3 --folder 29  # Coop, c=0.5
python plot_local_sx_pid_per_trial.py --path "${LOADPATH}" --trials 3 --folder 28  # Social, c=0.5
python plot_local_sx_pid_per_trial.py --path "${LOADPATH}" --trials 3 --folder 53  # Social, c=1.0

# ------------------------------------------------------ Logic Gate Simulations
python compare_correlated_logic_gates.py

# ----------------------------------------------------------- Methods/Appenddix
python plot_probability_distributions.py
python export_avg_local_sx_pid.py --coop_params 0.0 0.5 1.0
