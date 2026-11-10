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
col_unq_s1 = "unq1"
col_unq_s2 = "unq2"
col_coinfo = "coinfo"
col_syn_norm = "syn_norm"
col_shd_norm = "shd_norm"
col_unq_s1_norm = "unq1_norm"
col_unq_s2_norm = "unq2_norm"
col_coinfo_norm = "coinfo_norm"

# Settings file for plotting LBF results
plot_elements = {
    "max_width_in": 7.0,  # max. Figure width for Nature is 180mm = 7.08661 Inch
    "row_height": 1.6,
    "marker_size": 1,
    "linewidth": 1,
    "font_size": 7,
    "min_font_size": 5,
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
    "total_food_value_collected": r"$F$",
    "n_collections": "N collections",
    "n_collections_agent_0": r"$G_{A_1}$",  # "N collections $A_1$",
    "n_collections_agent_1": r"$G_{A_2}$",  # "N collections $A_2$",
    "food_value_collected_agent_0": r"$F_{A_1}$",
    "food_value_collected_agent_1": r"$F_{A_2}$",
    "any_food_collected": r"$G$",  # "any food collected",
    "any_food": "# collected",
    "food_type": "food type collected",
    "frac_cooperative_actions": "% joint actions",
    "n_cooperative_actions": r"$J$",
    "p_joint_action": r"$p(A_1=1,A_2=1,G=1)$",
    "local_shd_joint_action": r"$p*i_{shd}(G=1;A_1=1,A_2=1)$",
    "p_indiv1_action": r"$p(A_1=1,A_2=0,G=1)$",
    "local_unq1_indiv_action": r"$p*i_{unq1}(G=1;A_1=1,A_2=0)$",
    # Info-theoretic measures
    "h": r"$H(G)$",
    "mi_sources": r"$I(A_1;A_2)$",
    "jmi": r"$I(G;A_1,A_2)$",
    "source_corr": r"$r(A_1,A_2)$",
    "cmi_01": r"$I(A_1;F|A_2)$",
    "cmi_10": r"$I(A_2;F|A_1)$",
    "unq_own_norm_sx_cor": r"$I_{unq}^{SxPID}(G^{A_i};A_i)/I(G; A_1,A_2)$",
    "unq_other_norm_sx_cor": r"$I_{unq}^{SxPID}(G^{A_i};A_j)/I(G; A_1,A_2)$",
}

labels_short = labels.copy()
for norm, norm_label in zip(["", "norm_"], ["", "/I(G; A_1,A_2)"]):
    for estimator in ["sx", "sx_cor", "iccs", "broja", "dep"]:
        if estimator == "sx_cor":
            estimator_label = "nn"
        else:
            estimator_label = estimator
        pid_label = "(G;A_1,A_2)"
        labels[f"{norm}{estimator}"] = (
            r"$I^{" + estimator_label + "}" + pid_label + norm_label + "$"
        )
        labels_short[f"{norm}{estimator}"] = (
            r"$I^{" + estimator_label + "}" + norm_label[:2] + "$"
        )
        for atom in [col_syn, col_shd, col_unq_s1, col_unq_s2, col_mi]:
            atom_label = atom
            pid_label = "(G;A_1,A_2)"
            if atom == col_unq_s1:
                pid_label = "(G;A_1)"
                atom_label = "unq"
            elif atom == col_unq_s2:
                pid_label = "(G;A_2)"
                atom_label = "unq"

            labels[f"{atom}_{norm}{estimator}"] = (
                r"$I_{"
                + atom_label
                + "}^{"
                + estimator_label
                + "}"
                + pid_label
                + norm_label
                + "$"
            )
            labels[atom] = r"$I_{" + atom_label + "}" + pid_label + "$"
            labels_short[rf"{atom}_{norm}{estimator}"] = (
                r"$I_{"
                + atom_label
                + "}^{"
                + estimator_label
                + "}"
                + norm_label[:2]
                + "$"
            )

        labels[rf"coinfo_{norm}{estimator}"] = (
            "$coI^{" + estimator + "}(G;A_1,A_2)" + norm_label + "$"
        )
        labels_short[rf"coinfo_{norm}{estimator}"] = (
            "$coI^{" + estimator + "}" + norm_label[:2] + "$"
        )
