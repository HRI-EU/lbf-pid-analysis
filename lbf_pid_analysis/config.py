#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Config file -- specifies variables that are used over modules.
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
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
actions = {
    "Action.NONE": 0,
    "Action.NORTH": 1,
    "Action.SOUTH": 2,
    "Action.WEST": 3,
    "Action.EAST": 4,
    "Action.LOAD": 5,
}
sx_pid_lattice = {
    "unq1": ((1,),),
    "unq2": ((2,),),
    "shd": ((1,), (2,)),
    "syn": ((1, 2),),
}
i_ccs_lattice = {
    "unq1": ((0,),),
    "unq2": ((1,),),
    "shd": ((0,), (1,)),
    "syn": ((0, 1),),
}
syndisc_lattice = {  # opposed to the other lattices, this is a syn not a red/shd lattice, so the nodes differ
    "unq1": ((1,),),
    "unq2": ((0,),),
    "shd": (),
    "syn": ((0,), (1,)),
}
colors = {
    "red_muted": "#af363c",
    "red_muted_lighter_1": "#cf868a",
    "red_muted_lighter_2": "#e7c2c4",
    "blue_muted": "#003653",
    "blue_muted_lighter_1": "#668697",
    "blue_muted_lighter_2": "#99aeba",
    "green_muted": "#246920",
    "green_muted_lighter_1": "#659662",
    "green_muted_lighter_2": "#a7c3a5",
    "grey_muted": "#3F5661",
    "grey_muted_lighter_1": "#8b99a0",
    "grey_muted_lighter_2": "#c5cccf",
    "red": "#DB444B",
    "blue": "#006BA2",
    "cyan": "#3EBCD2",
    "green": "#379A8B",
    "yellow": "#EBB434",
    "purple": "#9A607F",
    "red_light_2": "#FF8785",
    "blue_light_2": "#7BBFFC",
    "cyan_light_2": "#6fe4fb",
    "green_light_2": "#69C9B9",
    "blue_dark_0": "#003653ff",
    "blue_dark_1": "#052438ff",
    "orange": "#f15b26ff",
    "gray": "#bebebe",  # "gray",
    "gray_light": "#b5b7baff",
    "gray_light_1": "#d8d8d8",  # "#999999",
    "gray_light_2": "#e6e6e6",  # "#afafaf",
    "gray_dark": "#969696",  # "dimgray",
    "violet": "#b1509eff",
    "bluegray": "#758D99",
    "bluegray_dark": "#3F5661",
    "bluegray_light_1": "#89A2AE",
    "bluegray_light_2": "#A4BDC9",
    "bluegray_2": "#bfc4cbff",
    "bluegray_3": "#6b7583ff",
    "bluegray_4": "#474e57ff",
    "histogram": "#6db5d1",
    "histogram_dark": "#569fba",
    "histogram_data": "xkcd:salmon",
}

col_s1 = "s1"
col_s2 = "s2"
col_t = "t"
col_count = "count"
col_joint_prob = "p(s1,s2,t)"
col_s1_label = "s1_label"
col_s2_label = "s2_label"
col_t_label = "t_label"
col_lmi_s1_t = "i(s1;t)"
col_lmi_s2_t = "i(s2;t)"
col_lmi_s1_s2_t = "i(s1,s2;t)"
col_mi = "mi"
col_syn = "syn"
col_shd = "shd"
col_unq_s1 = "unq_s1"
col_unq_s2 = "unq_s2"
col_coinfo = "coinfo"
col_coop_ratio = "coop_ratio"
col_syn_norm = "syn_norm"
col_shd_norm = "shd_norm"
col_unq_s1_norm = "unq_s1_norm"
col_unq_s2_norm = "unq_s2_norm"
col_coinfo_norm = "coinfo_norm"
col_coop_ratio_norm = "coop_ratio_norm"

# Comparisons performed for the estimated quantities
statistical_tests = {
    "comparisons": [
        ["MH_BASELINE", "MH_SOCIAL1"],
        ["MH_BASELINE", "MH_COOPERATIVE"],
        ["MH_BASELINE", "MH_ADAPTIVE"],
        ["MH_EGOISTIC", "MH_SOCIAL1"],
        ["MH_EGOISTIC", "MH_COOPERATIVE"],
        ["MH_EGOISTIC", "MH_ADAPTIVE"],
        ["MH_ADAPTIVE", "MH_COOPERATIVE"],
        ["MH_SOCIAL1", "MH_COOPERATIVE"],
    ],
    "dependent_variables": [
        ["syn_norm_sx_cor", "any_food"],
        ["syn_norm_sx_cor", "n_collections_agent_0"],
        "total_food_value_collected",
        "n_cooperative_actions",
    ],
}

# Settings file for plotting LBF results


plot_elements = {
    "figsize": {
        "maxwidth_cm": 12,  # \textwidth of standard LaTeX articles
        "maxwidth_in": 4.77,  # \textwidth of standard LaTeX articles
        "lineheight_in": 1.3,
        "colwidth_in": 0.954,  # max. width divided by 5
    },
    "marker_size": 1,
    "linewidth": 1,
    "box": {"linewidth": 0.7, "boxwidth": 0.6},
    "fontfamily": "sans-serif",  # 'cursive', 'fantasy', 'monospace', 'sans', 'sans serif', 'sans-serif', 'serif',
    "textsize": {  # Nature recommended sizes
        "small": 5,
        "medium": 6,
        "large": 8,
    },
    "scatter": {
        "size": 15,
        "marker": "o",
        "alpha": 0.7,
        "edgecolors": "w",
        "linewidth": 0.5,
    },
}

labels = {  # x- and y-axis labels for plots
    # Heuristics
    "MH_BASELINE": "BL",
    "MH_EGOISTIC": "Ego",  # former H1
    "MH_SOCIAL1": "Social",  # former H2
    "MH_SOCIAL2": "SOC2",  # former H5
    "MH_COOPERATIVE": "Coop",  # value function, coop
    "MH_ADAPTIVE": "Adapt",  # value function, coop or egoistic
    "MH_COMPETITIVE_EGOISTIC": "COMP_EGO",
    "MH_COMPETITIVE_OPPORTUNISTIC": "COMP_OPP",
    # Target variables
    "mean_food_value_collected": "sum. mean value collected",
    "total_food_value_collected": "$F$",
    "n_collections": "N collections",
    "n_collections_agent_0": "$G_{A_0}$",  # "N collections $A_0$",
    "n_collections_agent_1": "$G_{A_1}$",  # "N collections $A_1$",
    "food_value_collected_agent_0": "$F_{A_0}$",
    "food_value_collected_agent_1": "$F_{A_1}$",
    "any_food_collected": "$G$",  # "any food collected",
    "any_food": "# collected",
    "food_type": "food type collected",
    "frac_cooperative_actions": "% joint actions",
    "n_cooperative_actions": "$J$",
    "p_joint_action": "$p(A_0=1,A_1=1,G=1)$",
    "local_shd_joint_action": "$p*i_{shd}(G=1;A_0=1,A_1=1)$",
    "p_indiv1_action": "$p(A_0=1,A_1=0,G=1)$",
    "local_unq1_indiv_action": "$p*i_{unq1}(G=1;A_0=1,A_1=0)$",
    # Info-theoretic measures
    "h": "$H(F)$",
    "mi_sources": "$I(A_0;A_1)$",
    "mi": "$I(G;A_0,A_1)$",
    "source_corr": "$r(A_0,A_1)$",
    "cmi_01": "$I(A_0;F|A_1)$",
    "cmi_10": "$I(A_1;F|A_0)$",
    "unq_own_norm_sx_cor": "$I_{unq}^{SxPID}(G^{A_i};A_i)/I(G; A_0,A_1)$",
    "unq_other_norm_sx_cor": "$I_{unq}^{SxPID}(G^{A_i};A_j)/I(G; A_0,A_1)$",
    # # PID measures
    # "coinfo_sx": "$coI^{SxPID}(G;A_0,A_1)$",
    # "coinfo_norm_sx": "$coI^{SxPID}(G;A_0,A_1)/I(G;A_0,A_1)$",
    # "coop_ratio_sx": "$I^{SxPID}_{shd}(G;A_0,A_1)/I^{SxPID}_{syn}(G;A_0,A_1)$",
    # "coop_ratio_norm_sx": "$(I^{SxPID}_{shd}(G;A_0,A_1)/I^{SxPID}_{syn}(G;A_0,A_1))/I$",
    # "coinfo_broja": "$coI^{BROJA}(G;A_0,A_1)$",
    # "coinfo_norm_broja": "$coI^{BROJA}(G;A_0,A_1)$",
    # "coop_ratio_broja": "$I^{BROJA}_{shd}(G;A_0,A_1)/I^{BROJA}_{syn}(G;A_0,A_1)$",
    # "coop_ratio_norm_broja": "$(I^{BROJA}_{shd}(G;A_0,A_1)/I^{BROJA}_{syn}(G;A_0,A_1))/I$",
    # "syn_sx": "$I_{syn}^{SxPID}(G;A_0,A_1)$",
    # "syn_norm_sx": "$I_{syn}^{SxPID}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "shd_sx": "$I_{shd}^{SxPID}(G;A_0,A_1)$",
    # "shd_norm_sx": "$I_{shd}^{SxPID}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "unq1_sx": "$I_{unq}^{SxPID}(G;A_0)$",
    # "unq1_norm_sx": "$I_{unq}^{SxPID}(G;A_0)/I(G; A_0,A_1)$",
    # "unq2_sx": "$I_{unq}^{SxPID}(G;A_1)$",
    # "unq2_norm_sx": "$I_{unq}^{SxPID}(G;A_1)/I(G; A_0,A_1)$",
    # "mi_sx": "$I^{SxPID}(G;A_0,A_1)$",
    # "mi_norm_sx": "$I^{SxPID}(G;A_0,A_1)/H(G)$",
    # "coinfo_sx_cor": "$coI^{SxPID}(G;A_0,A_1)$",
    # "coinfo_norm_sx_cor": "$coI^{SxPID}(G;A_0,A_1)/I(G;A_0,A_1)$",
    # "coop_ratio_sx_cor": "$I^{SxPID}_{shd}(G;A_0,A_1)/I^{SxPID}_{syn}(G;A_0,A_1)$",
    # "coop_ratio_norm_sx_cor": "$(I^{SxPID}_{shd}(G;A_0,A_1)/I^{SxPID}_{syn}(G;A_0,A_1))/I$",
    # "syn_sx_cor": "$I_{syn}^{SxPID}(G;A_0,A_1)$",
    # "syn_norm_sx_cor": "$I_{syn}^{SxPID}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "shd_sx_cor": "$I_{shd}^{SxPID}(G;A_0,A_1)$",
    # "shd_norm_sx_cor": "$I_{shd}^{SxPID}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "unq1_sx_cor": "$I_{unq}^{SxPID}(G;A_0)$",
    # "unq1_norm_sx_cor": "$I_{unq}^{SxPID}(G;A_0)/I(G; A_0,A_1)$",
    # "unq2_sx_cor": "$I_{unq}^{SxPID}(G;A_1)$",
    # "unq2_norm_sx_cor": "$I_{unq}^{SxPID}(G;A_1)/I(G; A_0,A_1)$",
    # "mi_sx_cor": "$I^{SxPID}(G;A_0,A_1)$",
    # "mi_norm_sx_cor": "$I^{SxPID}(G;A_0,A_1)/H(G)$",
    # "syn_syndisc": "$I_{syn}^{SD}(G;A_0,A_1)$",
    # "syn_norm_syndisc": "$I_{syn}^{SD}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "shd_syndisc": "$I_{shd}^{SD}(G;A_0,A_1)$",
    # "shd_norm_syndisc": "$I_{shd}^{SD}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "unq1_syndisc": "$I_{unq}^{SD}(G;A_0)$",
    # "unq1_norm_syndisc": "$I_{unq}^{SD}(G;A_0)/I(G; A_0,A_1)$",
    # "unq2_syndisc": "$I_{unq}^{SD}(G;A_1)$",
    # "unq2_norm_syndisc": "$I_{unq}^{SD}(G;A_1)/I(G; A_0,A_1)$",
    # "mi_syndisc": "$I^{SD}(G;A_0,A_1)$",
    # "mi_norm_syndisc": "$I^{SD}(G;A_0,A_1)/H(G)$",
    # "syn_iccs": "$I_{syn}^{CCS}(G;A_0,A_1)$",
    # "syn_norm_iccs": "$I_{syn}^{CCS}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "shd_iccs": "$I_{shd}^{CCS}(G;A_0,A_1)$",
    # "shd_norm_iccs": "$I_{shd}^{CCS}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "unq1_iccs": "$I_{unq}^{CCS}(G;A_0)$",
    # "unq1_norm_iccs": "$I_{unq}^{CCS}(G;A_0)/I(G; A_0,A_1)$",
    # "unq2_iccs": "$I_{unq}^{CCS}(G;A_1)$",
    # "unq2_norm_iccs": "$I_{unq}^{CCS}(G;A_1)/I(G; A_0,A_1)$",
    # "mi_iccs": "$I^{CCS}(G;A_0,A_1)$",
    # "mi_norm_iccs": "$I^{CCS}(G;A_0,A_1)/H(G)$",
    # "syn_broja": "$I_{syn}^{BROJA}(G;A_0,A_1)$",
    # "syn_norm_broja": "$I_{syn}^{BROJA}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "shd_broja": "$I_{shd}^{BROJA}(G;A_0,A_1)$",
    # "shd_norm_broja": "$I_{shd}^{BROJA}(G;A_0,A_1)/I(G; A_0,A_1)$",
    # "unq1_broja": "$I_{unq}^{BROJA}(G;A_0)$",
    # "unq1_norm_broja": "$I_{unq}^{BROJA}(G;A_0)/I(G; A_0,A_1)$",
    # "unq2_broja": "$I_{unq}^{BROJA}(G;A_1)$",
    # "unq2_norm_broja": "$I_{unq}^{BROJA}(G;A_1)/I(G; A_0,A_1)$",
    # "mi_broja": "$I^{BROJA}(G;A_0,A_1)$",
    # "mi_norm_broja": "$I^{BROJA}(G;A_0,A_1)/H(G)$",
}

labels_short = labels.copy()
for norm, norm_label in zip(["", "_norm"], ["", "/I(G; A_0,A_1)"]):
    for estimator in ["sx", "sx_cor", "iccs", "broja", "syndisc"]:
        for atom in ["syn", "shd", "unq1", "unq2", "mi"]:
            if atom == "unq1":
                atom_label = "unq1"
                pid_label = "(G;A_0)"
            elif atom == "unq2":
                atom_label = "unq2"
                pid_label = "(G;A_1)"
            else:
                atom_label = atom
                pid_label = "(G;A_0,A_1)"
            if estimator == "sx_cor":
                estimator_label = "sx_{cor}"
            else:
                estimator_label = estimator
            labels[f"{atom}{norm}_{estimator}"] = (
                "$I_{"
                + atom_label
                + "}^{"
                + estimator_label
                + "}"
                + pid_label
                + norm_label
                + "$"
            )
            labels_short[f"{atom}{norm}_{estimator}"] = (
                "$I_{"
                + atom_label
                + "}^{"
                + estimator_label
                + "}"
                + norm_label[:2]
                + "$"
            )

        labels[f"coinfo{norm}_{estimator}"] = (
            "$coI^{" + estimator + "}(G;A_0,A_1)" + norm_label + "$"
        )
        labels[f"coop_ratio{norm}_{estimator}"] = (
            "$I^{"
            + estimator
            + "}_{shd}(G;A_0,A_1)/I^{"
            + estimator
            + "}_{syn}(G;A_0,A_1)"
            + norm_label
            + "$"
        )
        labels_short[f"coinfo{norm}_{estimator}"] = (
            "$coI^{" + estimator + "}" + norm_label[:2] + "$"
        )
        labels_short[f"coop_ratio{norm}_{estimator}"] = (
            "$I^{"
            + estimator
            + "}_{shd}/I^{"
            + estimator
            + "}_{syn}"
            + norm_label[:2]
            + "$"
        )
