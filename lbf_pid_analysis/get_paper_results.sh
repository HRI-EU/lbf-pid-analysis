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
LOADPATH="../../lbf_experiments/shared_goal_dist_0_0_v9/"

for TARGET in any_food total_food_value_collected n_collections food_type n_collections_agent_0
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 3 4 5 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 6 7 8 9 10 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 11 12 13 14 15 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 16 17 18 19 20 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 21 22 23 24 25 --target "${TARGET}" --path "${LOADPATH}"
done

python compare_pid_between_heuristics_bayes.py -f 1 2 3 4 5 --path "${LOADPATH}"
python compare_pid_between_heuristics_bayes.py -f  6 7 8 9 10 --path "${LOADPATH}"
python compare_pid_between_heuristics_bayes.py -f 11 12 13 14 15 --path "${LOADPATH}"
python compare_pid_between_heuristics_bayes.py -f 16 17 18 19 20 --path "${LOADPATH}"
python compare_pid_between_heuristics_bayes.py -f  21 22 23 24 25 --path "${LOADPATH}"
python summarize_group_comparisons.py --path "${LOADPATH}" --stats_type bayes --measure syn_norm_sx_cor_any_food
python summarize_group_comparisons.py --path "${LOADPATH}" --stats_type bayes --measure syn_norm_sx_cor_n_collections_agent_0

for TARGET in any_food total_food_value_collected n_collections food_type n_collections_agent_0
do
    python compare_correlation_between_heuristics.py -m syn_norm_sx --target "${TARGET}" --path "${LOADPATH}"
    python compare_correlation_between_heuristics.py -m syn_norm_sx_cor --target "${TARGET}" --path "${LOADPATH}"
    python plot_measure_by_c.py --path "${LOADPATH}" -m syn_norm_iccs --target "${TARGET}"
    python plot_measure_by_c.py --path "${LOADPATH}" -m syn_norm_sx --target "${TARGET}"
    python plot_measure_by_c.py --path "${LOADPATH}" -m syn_norm_sx_cor --target "${TARGET}"
    python plot_measure_by_c.py --path "${LOADPATH}" -m syn_norm_syndisc --target "${TARGET}"
done

python plot_action_type.py --path "${LOADPATH}"
python compare_synergy_and_task_success.py --path "${LOADPATH}" -t n_collections_agent_0 --success total_food_value_collected
python compare_synergy_and_task_success.py --path "${LOADPATH}" -t any_food --success total_food_value_collected

for TARGET in any_food n_collections total_food_value_collected
do
    echo "${TARGET}"
    for FOLDER in {1..25}
    do
        python plot_local_sx_pid_per_trial.py -t "${TARGET}" -f "${FOLDER}" --path "${LOADPATH}"
    done
done

# ------------------------------------------------------------- dist = 0.2
LOADPATH="../../lbf_experiments/shared_goal_dist_0_2_v9/"

for TARGET in any_food total_food_value_collected n_collections food_type
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 3 4 5  --target "${TARGET}" --path "${LOADPATH}"
done

python compare_pid_between_heuristics_bayes.py -f 1 2 3 4 5 --path "${LOADPATH}"
python summarize_group_comparisons.py --path "${LOADPATH}" --stats_type bayes --measure syn_norm_sx_cor

# ------------------------------------------------------------- dist = 0.5
LOADPATH="../../lbf_experiments/shared_goal_dist_0_5_v9/"

for TARGET in any_food
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 3 4 5 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 6 7 8 9 10 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 11 12 13 14 15 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 16 17 18 19 20 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 21 22 23 24 25 --target "${TARGET}" --path "${LOADPATH}"
done

# ------------------------------------------------------------- asymmetric
LOADPATH="../../lbf_experiments/asymmetric_d_0_0_v9/"

for TARGET in any_food n_collections_agent_0 n_collections_agent_1
do
    echo "${TARGET}"
    python analyze_pid_per_trial.py -f 1 2 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 3 4 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 5 6 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 7 8 --target "${TARGET}" --path "${LOADPATH}"
    python analyze_pid_per_trial.py -f 9 10 --target "${TARGET}" --path "${LOADPATH}"
done

python compare_synergy_asymmetric_cond.py --path "${LOADPATH}"

# ------------------------------------------------------------- plots
python plot_full_paper_figures.py --version 9
