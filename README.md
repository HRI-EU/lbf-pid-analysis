# Introduction

This repository contains scripts to analyze game data generated with HRI-EU's fork of the level-based foraging
environment (LBF). Analysis comprises the estimation of information-theoretic measures, in particular, partial
information decomposition (PID), between agents' actions and the amount of food items collected.

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

Set up the environment via

```bash
uv sync
```

Then, scripts can be run within the `lbf_pid_analysis` subfolder. To replicate analyses shown in the paper, run the
LBF experiment. Then run `run_pid_estimation_for_paper.sh` and generate plots via `paper_figures.ipynb` after updating
the path to the experimental data in both files.

Running analyses presented in this repository assumes game data generated with HRI-EU's fork of the LBF
environment: [https://github.com/HRI-EU/lb-foraging](https://github.com/HRI-EU/lb-foraging). See the repository's README
for details on running experiments.

## Environment Setup and Usage

To install dependencies and run the main analysis file, do the following once to prepare the Python environment:

```bash
$ uv sync
```

To activate the virtual environment, run

```bash
source .venv/bin/activate
```

in the top-level folder.

Note that the following non-python dependencies may have to be installed additionally:

```bash
openjdk 17  # JDK 17
```

## Running PID estimation

To estimate PID, call

```bash
python analyze_pid_per_trial.py --folders 1 2 3 4 5 --path ../../lbf_results/experiment --target 'any_food'
```

where the path is the output path defined in the settings file and folders is a list of folders to compare (e.g., 3
different heuristics over one value of $c$). The `target` parameter is explained in detail in the next section.

## Running full analysis from paper

To run all analyses as reported in the paper and generate figures from results, run

```bash
cd lbf_pid_analysis
./run_pid_estimation_for_paper.sh
```

after updating the path to the experimental results in the script.

If using uv, run:

```bash
cd lbf_pid_analysis
uv run bash ./run_pid_estimation_for_paper.sh
```

Generate plots with `paper_figures.ipynb` (again, update the path to experimental results in the notebook).

The scripts assume that game data, simulated with the LBF environment are available under
`"../../lbf_experiment_results/"`.

# Settings for estimating information-theoretic measures from LBF agent behavior

## Target variable encodings for PID estimation

The `target` parameter defines which target variable to use for PID estimation. The following variables are implemented:

- `any_food`: 1 if any food item was collected by any agent in a time step, 0 otherwise (nothing collected)
- `cooperative_actions`: 1 if a food item was collected collectively, 0 otherwise
- `total_food_value_collected`: total food value collected in each time step, 0 otherwise (nothing collected)
- `mean_food_value_collected`: mean food value collected in each time step, 0 otherwise (nothing collected)
- `food_type`: 2 if a cooperative/"heavy" food item was collected collectively in a time step, 1 if either agent collected
  a non-cooperative/"light" item, 0 otherwise (nothing collected)
- `n_collections`: number of food items collected in each step (0-2), the difference to variable 'food_type' is that this
  can also be two if two agents collect a food item individually in the same time step
- `n_collections_agent_0`: number of food items collected in each step by agent 0 (0 or 1)
- `n_collections_agent_1`: number of food items collected in each step by agent 1 (0 or 1)

## Source variable encodings for PID estimation

Which sources are used in PID estimation and how they are encoded is specified in the settings file
(`config/settings.yml`). There are three options for choosing source variables:

- `analysis:sources: 'actions'`: use an agent's action in each time step
    - `analysis:source_encoding: 'binary'`: encode actions as binary variables: 1 for action LOAD, 0 otherwise
    - `analysis:source_encoding: 'individual_actions'`: encode individual actions from 0 = none to 5 = load
- `analysis:sources: 'closest_distance'`: uses the distance to the closest food item in each time step

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
@article{
  title={Beyond Performance: An Information-Theoretic Account of Behavioral Organization in Cooperative Tasks}.
  author={Wollstadt, Patricia and Makkeh, Abdullah and Kr\"uger, Matti and Wiebel-Herboth, Christiane},
  year={2026}
}
```

# Related repositories

- Original LBF Project Link: [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging) by
  Filippos Christianos (f.christianos@ed.ac.uk)
- HRI-EU's LBF fork that generates game data for analysis and implements additional cooperative agent
  heuristics: [https://github.com/HRI-EU/lb-foraging](https://github.com/HRI-EU/lb-foraging) by
    - Matti Krüger (matti.krueger@honda-ri.de)
    - Christiane Wiebel-Herboth (christiane.wiebel@honda-ri.de)
    - Patricia Wollstadt (patricia.wollstadt@honda-ri.de)

# Contact

Patricia Wollstadt - patricia.wollstadt@honda-ri.de - HRI-EU
