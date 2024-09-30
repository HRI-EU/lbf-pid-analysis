# Introduction

This repository contains scripts to analyze game data generated with HRI-EU's fork of
the level-based foraging environment (LBF). Analysis comprises the estimation of
information-theoretic measures, in particular, partial information decomposition (PID),
between agents' actions and the amount of food items collected.

# Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Estimating information-theoretic measures from LBF agent behavior](#estimating-information-theoretic-measures-from-lbf-agent-behavior)
  - [Running PID estimation](#running-pid-estimation)
  - [Target variable encodings for PID estimation](#target-variable-encodings-for-pid-estimation)
  - [Source variable encodings for PID estimation](#source-variable-encodings-for-pid-estimation)
- [Please Cite](#please-cite)
- [Related repositories](#related-repositories)
  - [Contact](#contact)


<!-- GETTING STARTED -->
# Getting Started

Scripts can be run within the `lbf_pid_analysis` subfolder. To replicate analyses
shown in the paper, run `get_paper_results.sh` and `plot_paper_figures.sh`.

Running analyses presented in this repository assumes game data generated with HRI-EU's
fork of the LBF environment: [https://github.com/HRI-EU/lb-foraging](https://github.com/HRI-EU/lb-foraging).
See the repository's README for details on running experiments.

# Estimating information-theoretic measures from LBF agent behavior

## Running PID estimation

To estimate PID, call

```bash
python analyze_pid_per_trial.py --folders 1 2 3 4 5 --path ../../lbf_results/experiment --target 'any_food'
```

where the path is the output path defined in the settings file and folders is a list of folders to compare (e.g., 3 different heuristics over one value of $c$). The `target` parameter is explained in detail in the next section.

## Target variable encodings for PID estimation

The `target` parameter defines which target variable to use for PID estimation. The following variables are implemented:

- 'cooperative_actions': 1 if a food item was collected collectively, 0 otherwise
- total_food_value_collected: total food value collected in each time step, 0 otherwise (nothing collected)
- mean_food_value_collected: mean food value collected in each time step, 0 otherwise (nothing collected)
- any_food: 1 if any food item was collected by any agent in a time step, 0 otherwise (nothing collected)
- food_type: 2 if a cooperative/"heavy" food item was collected collectively in a time step, 1 if either agent collected a non-cooperative/"light" item, 0 otherwise (nothing collected)
- n_collections: number of food items collected in each step (0-2), the difference to variable 'food_type' is that this can also be two if two agents collect a food item individually in the same time step
- n_collections_agent_0: number of food items collected in each step by agent 0 (0 or 1)
- n_collections_agent_1: number of food items collected in each step by agent 1 (0 or 1)

## Source variable encodings for PID estimation

Which sources are used in PID estimation and how they are encoded is specified in the settings file (`config/settings.yml`). There are three options for choosing source variables:

- `analysis:sources: 'closest_distance'`: uses the distance to the closest food item in each time step
- `analysis:sources: 'actions'`: use an agent's action in each time step
  - `analysis:source_encoding: 'binary'`: encode actions as binary variables: 1 for action LOAD, 0 otherwise
  - `analysis:source_encoding: 'individual_actions'`: encode individual actions from 0 = none to 5 = load


# Please Cite
1. The original paper that introduces the LBF environment for reinforcement learning:
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
2. The paper that presents PID estimation for cooperative agent behavior:
```
TODO
```
# Related repositories


- Original LBF Project Link: [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging) by Filippos Christianos (f.christianos@ed.ac.uk)
- HRI-EU's LBF fork that generates game data for analysis and implements additional cooperative
agent heuristics: [https://github.com/HRI-EU/lb-foraging](https://github.com/HRI-EU/lb-foraging) by
  - Matti Krüger (matti.krueger@honda-ri.de)
  - Christiane Wiebel-Herboth (christiane.wiebel@honda-ri.de)
  - Patricia Wollstadt (patricia.wollstadt@honda-ri.de)

## Contact

Patricia Wollstadt - patricia.wollstadt@honda-ri.de  - HRI-EU
